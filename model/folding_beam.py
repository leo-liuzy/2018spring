import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys

SMALLEST_LOG = -sys.float_info.max


class FoldingBeam(object):
    """
    This beam uses tensors to store value so it's compatible with batch process
    """

    def __init__(self,
                 batch_size: int,
                 beam_width: int,
                 end_index: int,
                 decoder_output_dim: int, stop_index: str):
        """

        :param batch_size:
        :param beam_width:
        :param end_index:
        """
        self._batch_size = batch_size
        self._beam_width = beam_width
        self._end_index = end_index
        self._decoder_output_dim = decoder_output_dim

        # batched sequence of indices; the dimension will change
        # decoded_seq and decoded_log_prob correspond to each other, we will keep appending
        # to each list to keep the order unchanged
        self.decoded_seq = {i: [] for i in range(self._batch_size)}
        self.decoded_log_prob = {i: [] for i in range(self._batch_size)}
        self.sequences = Variable(torch.LongTensor(self._batch_size, self._beam_width).zero_())

        # batched log prob of the sequences; the following tensor will have their dimension fixed
        self.seq_log_prob = Variable(torch.zeros(self._batch_size, self._beam_width, 1))
        self.hiddens = Variable(torch.zeros(self._batch_size, self._beam_width, self._decoder_output_dim))
        self.contexts = Variable(torch.zeros(self._batch_size, self._beam_width, self._decoder_output_dim))

    def batched_index_select(self, batched_input, batched_indices: torch.LongTensor):
        """

        :param batched_input:  (batch_size, beam_width, cur_seq_len)
        :param batched_indices: (batch_size, beam_width, 1)
        :return:
        """
        batch_size, _, _ = batched_input.size()
        result = []
        for k in range(batch_size):
            sub_input = batched_input[k]
            sub_indices = batched_indices[k].view(-1)
            selected_result = torch.index_select(sub_input, 0, Variable(sub_indices))
            result.append(selected_result)

        return torch.stack(result, 0)

    def fold(self, class_probabilities, num_classes):
        """
        This function is called everytime we update the beam.
        It will fold finished sentence and then take the topk over all the rest, util there is no
        more finished sentences among the k options.
        """

        # the follwing three tensor is to mask
        stop_tensor = torch.LongTensor(self._beam_width, 1).fill_(self._end_index)
        one_tensor = torch.ByteTensor(self._beam_width, 1).fill_(1)
        # TODO: if want to change to a batched solution, probably need padding
        batch_next_indices = []
        batch_next_log_prob = []
        for i in range(self._batch_size):
            # loop through each batch
            stop = False
            cur_batch = class_probabilities[i]
            cur_batch_seq = self.sequences[i]
            cur_batch_log = self.seq_log_prob[i]
            # keep topk unless there is no finished sentences in that batch
            while not stop:

                log_prob_cur, indices = torch.topk(cur_batch, self._beam_width, dim=0)
                log_prob_cur = log_prob_cur.view(-1, 1)
                indices = indices.view(-1, 1)
                nth_class_per_beam = indices % num_classes  # (self._beam_width, 1)
                unfinished_tensor = nth_class_per_beam - stop_tensor != 0
                finished_tensor = nth_class_per_beam - stop_tensor == 0
                # dimension with value 1 means we predict a stop symbol

                if torch.equal(unfinished_tensor, one_tensor):
                    stop = True
                    # all the decoded sequence is not finished!
                    break
                else:
                    assert indices.size() == unfinished_tensor.size()
                    # mask the indices tensor
                    tmp = indices * unfinished_tensor
                    tmp_finished = indices * finished_tensor

                    # (x,)
                    unfinished_index = tmp[tmp.nonzero()[:, :0]].view(-1)
                    # (beam_width - x,)
                    finished_index = tmp_finished[tmp_finished.nonzero()[:, :0]].view(-1)

                    # TODO: seems like a bug
                    finished_seq = torch.index_select(cur_batch_seq, 0, finished_index)
                    finished_log_prob = torch.index_select(cur_batch_log, 0, finished_index)

                    # add the decoded seq and corresponding log_prob to the list
                    for seq in finished_seq:
                        self.decoded_seq[i].append(seq)
                    for log_prob in finished_log_prob:
                        self.decoded_log_prob[i].append(log_prob)

                    for index in finished_index:
                        # we find the finished decoding in the batch and set the log_prob
                        # to -sys.float_info.max so that it won't affect out next topk
                        # cur_batch = (beam_width, num_classes)
                        cur_batch[indices[index], :] = -sys.float_info.max

            assert log_prob_cur.size() == indices.size() and log_prob_cur.size() == (self._beam_width, 1)
            batch_next_indices.append(indices)
            batch_next_log_prob.append(log_prob_cur)

        batch_next_indices = torch.stack(batch_next_indices, 0)
        batch_next_log_prob = torch.stack(batch_next_log_prob, 0)
        assert len(batch_next_indices) == self._batch_size
        # print("batch_next_indices: " + str(batch_next_indices.size()))
        assert batch_next_indices.size(0) == self._batch_size and batch_next_indices.size(1) == self._beam_width

        return batch_next_indices, batch_next_log_prob

    def update(self, output_projections, hiddens, contexts):
        """
        update the current step of topk candidates values and previous normalized probability, given
        the new hidden outputs

        :param output_projections: projection of hidden states (batch_size, beam_width, num_classes)
        :param hiddens: one hidden vector for each beam (batch_size, beam_width, decoder_output_dim)
        :param contexts: one context vector for each beam  (batch_size, beam_width, decoder_output_dim)
        """

        # (batch, beam_width, num_classes)
        _, _, num_classes = output_projections.size()

        # (batch, beam_width * num_classes, 1)
        class_probabilities = F.softmax(output_projections, dim=-1)
        class_probabilities = torch.log(class_probabilities.view(self._batch_size, -1, 1))

        if len(self.sequences.size()) == 2:
            self.sequences = self.sequences.unsqueeze(2)  # (batch_size, beam_width, 1)

        # each return is (batch, beam_width, 1)
        # class_probabilities = (batch, beam_width * num_classes, 1)

        # prob_cur, indices = torch.topk(class_probabilities, self._beam_width, dim=1)
        update_indices, update_log_prob = self.fold(class_probabilities, num_classes)

        nth_beam = update_indices / num_classes  # used for selecting the hidden state and context vector
        update_seq = update_indices % num_classes

        assert self.sequences.size(0) == update_seq.size(0) and self.sequences.size(1) == update_seq.size(1)
        print("self.sequences size (before update): " + str(self.sequences.size()))
        self.sequences = torch.cat([self.sequences, update_seq], 2)
        print("self.sequences size: " + str(self.sequences.size()))
        assert self.seq_log_prob.size() == update_seq.size()
        self.seq_log_prob += update_log_prob

        # update the next hidden states and the context vector
        self.hiddens = self.batched_index_select(hiddens, nth_beam.data)
        self.contexts = self.batched_index_select(contexts, nth_beam.data)

    def get_last_prediction(self):

        assert self.sequences.size(0) == self._batch_size and self.sequences.size(1) == self._beam_width
        return self.sequences[:, :, -1].unsqueeze(2)

    def get_final_seq(self):
        # select the seq with the largest log prob in each batch
        _, batch_seq_indice = torch.max(self.seq_log_prob, 1)
        batch_seq_indice = batch_seq_indice.view(-1).data.numpy()
        result = []
        for i in range(self._batch_size):
            result.append(self.sequences[i, batch_seq_indice[i], :].data.view(1, -1))
        return Variable(torch.cat(result, 0))  # return value has the dimension (batch_size, seq_length)

    def get_n_best(self):
        """
        Pad and concatenate sentences and log_prob in the decoded and current storage
        :return:
        """
        pass
