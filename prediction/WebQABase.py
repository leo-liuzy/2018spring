from typing import Dict

import numpy
from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum
# from WebQA.utils.beam import Beam
# from .folding_beam import FoldingBeam
from .mmi_beam import MMIBeam

use_cuda = torch.cuda.is_available()


@Model.register("WebQABase")
class WebQABase(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    This ``SimpleSeq2Seq`` model takes an encoder (:class:`Seq2SeqEncoder`) as an input, and
    implements the functionality of the decoder.  In this implementation, the decoder uses the
    encoder's outputs in two ways. The hidden state of the decoder is initialized with the output
    from the final time-step of the encoder, and when using attention, a weighted average of the
    outputs from the encoder is concatenated to the inputs of the decoder at every timestep.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : int, required
        Length of decoded sequences
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention_function: ``SimilarityFunction``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    scheduled_sampling_ratio: float, optional (default = 0.0)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 mmi: dict,
                 beam_width: int,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 attention_function: SimilarityFunction = None,
                 scheduled_sampling_ratio: float = 0.0) -> None:
        super(WebQABase, self).__init__(vocab)
        self.mmi = mmi
        self.beam_width = beam_width
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._attention_function = attention_function
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder. Also, if
        # we're using attention with ``DotProductSimilarity``, this is needed.
        self._decoder_output_dim = self._encoder.get_output_dim()
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)
        if self._attention_function:
            self._decoder_attention = Attention(self._attention_function)
            # The output of attention, a weighted average over encoder outputs, will be
            # concatenated to the input vector of the decoder at each time step.
            self._decoder_input_dim = self._encoder.get_output_dim() + target_embedding_dim
        else:
            self._decoder_input_dim = target_embedding_dim
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        # print("source tokens: " + str(source_tokens))
        embedded_input = self._source_embedder(source_tokens)
        # embedded_input = embedded_input.cuda() if use_cuda else embedded_input
        batch_size, input_length, _ = embedded_input.size()
        source_mask = get_text_field_mask(source_tokens)
        # source_mask = source_mask.cuda() if use_cuda else source_mask
        print("batch_size: " + str(batch_size))

        # self._encoder = self._encoder.cuda() if use_cuda else self._encoder
        encoder_outputs = self._encoder(embedded_input, source_mask)
        # encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        print("encoder_outputs : " + str(encoder_outputs.size()))
        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        print("final_encoder_output: " + str(final_encoder_output.size()))
        print("target tokens : " + str(target_tokens))

        # self._decoder_cell = self._decoder_cell.cuda() if use_cuda else self._decoder_cell

        if target_tokens:
            targets = target_tokens["tokens"]
            target_sequence_length = targets.size(1)
            # The last input from the target is either padding or the end symbol. Either way, we
            # don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps
        beam_width = self.beam_width
        # in this decoding, we only use "folding" beam search

        if self.mmi["mode"] == "anti":
            beam = MMIBeam(batch_size=batch_size,
                           beam_width=beam_width,
                           end_index=self._end_index,
                           decoder_output_dim=self._decoder_output_dim,
                           lambd=self.mmi["lambda"])

            output_dict = self.mmi_antiLM(encoder_outputs=encoder_outputs,
                                          final_encoder_output=final_encoder_output,
                                          batch_size=batch_size,
                                          beam_width=beam_width,
                                          beam=beam,
                                          source_mask=get_text_field_mask(source_tokens),
                                          gamma=self.mmi["gamma"],
                                          max_decoding_step=num_decoding_steps)
        elif self.mmi["mode"] == "bidi":
            pass  # self.mmi_bidi(source_tokens, )
        else:
            assert False
        return output_dict

        # (batch, beam_width, decoder_output_dim)
        # here I unsqueeze a dimension of beam width, so that each beam and each batch can process in parallel

    def mmi_antiLM(self,
                   encoder_outputs,
                   final_encoder_output,
                   batch_size,
                   beam_width,
                   beam,
                   source_mask,
                   gamma,
                   max_decoding_step):
        """
        I use decoder with hidden state 0 to generate probability of response

        anti-LM is almost the same as normal beam search, but we choose value after penalizing the language model
        :param encoder_outputs:
        :param final_encoder_output:
        :param batch_size:
        :param beam_width:
        :param beam:
        :param source_mask:
        :param gamma:
        :param max_decoding_step:
        :return:
        """
        # (batch, beam_width, decoder_output_dim)
        # here I unsqueeze a dimension of beam width, so that each beam and each batch can process in parallel
        batched_encoder_outputs = torch.cat([encoder_outputs.data.unsqueeze(1)] * beam_width, 1)

        decoder_hiddens = Variable(torch.cat([final_encoder_output.data.unsqueeze(1)] * beam_width, 1))
        decoder_contexts = Variable(torch.zeros(batch_size, beam_width, self._decoder_output_dim))
        # decoder_contexts = decoder_contexts.cuda() if use_cuda else decoder_contexts

        assert decoder_hiddens.size() == decoder_contexts.size()
        print("decoder_hiddens.size: " + str(decoder_hiddens.size()))
        print("decoder_contexts.size: " + str(decoder_contexts.size()))

        lm_hiddens = Variable(torch.zeros(batch_size, beam_width, self._decoder_output_dim))
        lm_contexts = Variable(torch.zeros(batch_size, beam_width, self._decoder_output_dim))

        # lm_hiddens = lm_hiddens.cuda() if use_cuda else lm_hiddens
        # lm_contexts = lm_contexts.cuda() if use_cuda else lm_contexts

        print("lm_hiddens size: " + str(lm_hiddens.size()))
        print("lm_contexts size: " + str(lm_contexts.size()))
        print("-" * 50)
        print("Beam search starting....")
        # decoder_hidden (batch_size, beam_width, encoder_output_dim)
        # last_prediction (batch_size, beam_width, 1)
        last_predictions = None

        # batched_input_choices (batch_size, beam_width,) is the current input of the decoding step

        for timestep in range(max_decoding_step):  # loop through the batch size
            """
            if target_tokens and self.training and all(torch.rand(1) >= self._scheduled_sampling_ratio):
                # batch_size, beam_width
                # TODO: this is potentially a bug
                batched_input_choices = Variable(
                    torch.LongTensor(batch_size, beam_width, 1).fill_(targets[:, timestep]))
            else:"""
            if timestep == 0:
                # For the first timestep, when we do not have targets, we input start symbols.
                # (batch_size, beam_width,)

                batched_input_choices = Variable(
                    torch.LongTensor(batch_size, beam_width, 1).fill_(self._start_index))
                batched_lm_choices = Variable(
                    torch.LongTensor(batch_size, beam_width, 1).fill_(self._start_index))

                # batched_input_choices = batched_input_choices.cuda() if use_cuda else batched_input_choices
                # batched_lm_choices = batched_lm_choices.cuda() if use_cuda else batched_lm_choices


            else:
                batched_input_choices = last_predictions  # (batch_size, beam_width,)
                batched_lm_choices = last_predictions

            print("Batched input size: " + str(batched_input_choices.size()))
            print("Batched input: \n" + str(batched_input_choices))
            # input_choices is the index of the input in the embedding

            # decoder_input (batch_size, beam_width, _decoder_output_dim)
            decoder_input_cache = []
            lm_input_cache = []

            # assert batched_input_choices.is_same_size(decoder_hiddens)
            # prepare the input for the decoder(in batch)
            for i in range(batched_input_choices.size(0)):
                decoder_input = self._prepare_decode_step_input(batched_input_choices[i],
                                                                decoder_hiddens[i],
                                                                batched_encoder_outputs[i],
                                                                source_mask).squeeze(1)

                lm_input = self._prepare_decode_step_input(batched_lm_choices[i],
                                                           lm_hiddens[i],
                                                           batched_encoder_outputs[i],
                                                           source_mask).squeeze(1)
                decoder_input_cache.append(decoder_input)
                lm_input_cache.append(lm_input)

            decoder_input = torch.stack(decoder_input_cache, dim=0)
            lm_input = torch.stack(lm_input_cache, dim=0)

            hidden_cache = []
            context_cache = []
            lm_hidden_cache = []
            lm_context_cache = []
            # decode by batch
            for i in range(batch_size):
                # decoder_hidden/context -- (beam_width, _decoder_output_dim)
                assert decoder_hiddens[i].size() == decoder_contexts[i].size()
                decoder_hidden, decoder_context = self._decoder_cell(decoder_input[i],
                                                                     (decoder_hiddens[i], decoder_contexts[i]))
                lm_hidden, lm_context = self._decoder_cell(lm_input[i],
                                                           (lm_hiddens[i], lm_contexts[i]))

                hidden_cache.append(decoder_hidden)
                context_cache.append(decoder_context)

                lm_hidden_cache.append(lm_hidden)
                lm_context_cache.append(lm_context)

            decoder_contexts = torch.stack(context_cache, dim=0)
            decoder_hiddens = torch.stack(hidden_cache, dim=0)

            lm_hiddens = torch.stack(lm_hidden_cache, dim=0)
            lm_contexts = torch.stack(lm_context_cache, dim=0)

            # output_projections --- (batch_size, beam_width, num_classes)
            # get the final outout
            # soft-maxed projection
            output_projections = self._output_projection_layer(decoder_hiddens)
            lm_projections = self._output_projection_layer(lm_hiddens)
            g_k = 1 if timestep <= gamma else 0

            # use beam to update the N-best list
            beam.update(output_projections, decoder_hiddens, decoder_contexts,
                        lm_projections, lm_hiddens, lm_contexts, g_k)

            # (batch_size, longest_sequence_length)
            last_predictions = beam.get_last_prediction()

        # then we normalize the log_prob with the seq_length
        assert batch_size == 1
        prediction_list = beam.decoded_seq[0]
        print("len of prediction_list:")
        print(len(prediction_list))
        # print("prediction_list: ")
        # print(prediction_list)

        for i in range(len(prediction_list)):
            # print("decoded_log_prob[0][i]: " + str(beam.decoded_log_prob[0][i]))
            # print("prediction_list[i].size(1): " + str(prediction_list[i]))
            # assert len(prediction_list[i].size()) == 2 and prediction_list[i].size(0) == 1
            if len(prediction_list[i].size()) == 2:
                beam.decoded_log_prob[0][i] = beam.decoded_log_prob[0][i]/prediction_list[i].size(1)

        # seq (1, len_seq)
        for seq in beam.sequences[0]:
            prediction_list.append(seq)

        beam.seq_log_prob[0] /= beam.sequences.size(2)

        print("len(beam.decoded_log_prob[0]): ")
        print(len(beam.decoded_log_prob[0]))
        print(str(beam.decoded_log_prob[0]))
        if len(beam.decoded_log_prob[0]) > 0:
            log_prob_list = torch.cat(beam.decoded_log_prob[0], 0).view(-1, 1)
            log_prob_list = torch.cat([log_prob_list, beam.seq_log_prob[0]], dim=0)
        else:
            log_prob_list = torch.cat([beam.seq_log_prob[0]], dim=0)
        print("log_prob_list:" + str(log_prob_list))
        assert log_prob_list.size() == (len(beam.decoded_log_prob[0]) + len(beam.seq_log_prob[0]), 1)
        _, index = torch.max(log_prob_list, dim=0)
        print(index)
        final_prediction = prediction_list[index.data[0]].view(-1)
        print("final_prediction:")
        print(final_prediction)
        output_dict = {"final_predictions": final_prediction}
        print(output_dict)
        return output_dict

    def mmi_bidi(self, beam, lambd):
        # (batch_size, beam_width, max_length)
        batched_target_tokens = beam.sequences
        # (batch_size, beam_width, 1)

        batch_size, beam_width, _ = batched_target_tokens.size()
        target_embeddings = Variable(torch.zeros(batch_size, beam_width, ))

        # now we will condition on the target and generate the probability of source
        last_decode_output = None
        prev_hidden = None
        pre_context = None

        # no sure how to obtain the index of tokens in batched questions.
        _, batched_input = source_tokens

        generic_cost = []
        for i in range(batch_size):
            # first encode the target
            embedded_input = self._target_embedder(batched_target_tokens[i])
            target_mask = get_text_field_mask(batched_target_tokens[i])
            encoder_output = self._encoder(embedded_input, target_mask)
            final_encoder_output = encoder_outputs[:, -1]  # (beam_width, encoder_output_dim)
            log_prob = Variable(torch.zeros(beam_width, 1))
            expected_outputs = batched_input[i].view(-1, 1)

            # start generate source
            for j in range(input_length):
                if j == 0:
                    decoder_input = self._source_embedder(Variable(torch.zeros(beam_width, 1).fill_(self._start_index)))
                    decoder_hidden = final_encoder_output
                    decoder_context = torch.zeros(decoder_hidden.size())
                else:
                    decoder_input = last_decode_output
                    decoder_hidden = prev_hidden
                    decoder_context = pre_context

                decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                                     (decoder_hidden, decoder_context))

                expected_output = torch.ones(beam_width, 1).fill_(expected_outputs[j])
                output_projections = self._output_projection_layer(decoder_hiddens)
                class_probabilities = F.softmax(output_projections, dim=-1)
                log_prob = log_prob + class_probabilities.index_select(class_probabilities, 0, expected_output)

                last_decode_output = self._source_embedder(Variable(torch.zeros(beam_width, 1).fill_(expected_output)))
                prev_hidden = decoder_hidden
                pre_context = decoder_context
            generic_cost.append(log_prob)

        generic_cost = torch.stack(generic_cost, 0)
        return generic_cost


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["final_predictions"]  # seqs are in batch
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.data.cpu().numpy()
        # print('predicted_indices:' + str(predicted_indices))
        all_predicted_tokens = []
        predicted_tokens = [self.vocab.get_token_from_index(x, namespace="target_tokens")
                            for x in predicted_indices]

        all_predicted_tokens.append(" ".join(predicted_tokens))
        output_dict["final_predictions"] = all_predicted_tokens
        """
        for indices in predicted_indices:
            indices = indices.tolist()
            print("indices: " + str(indices))
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]

            predicted_tokens = [self.vocab.get_token_from_index(x, namespace="target_tokens")
                                for x in indices]

            all_predicted_tokens.append(" ".join(predicted_tokens))
            output_dict["final_predictions"] = all_predicted_tokens
        """
        return output_dict

    def _prepare_decode_step_input(self,
                                   input_indices: torch.LongTensor,
                                   decoder_hidden_state: torch.LongTensor = None,
                                   encoder_outputs: torch.LongTensor = None,
                                   encoder_outputs_mask: torch.LongTensor = None) -> torch.LongTensor:
        """
        Given the input indices for the current timestep of the decoder, and all the encoder
        outputs, compute the input at the current timestep.  Note: This method is agnostic to
        whether the indices are gold indices or the predictions made by the decoder at the last
        timestep. So, this can be used even if we're doing some kind of scheduled sampling.

        If we're not using attention, the output of this method is just an embedding of the input
        indices.  If we are, the output will be a concatentation of the embedding and an attended
        average of the encoder inputs.

        Parameters
        ----------
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the
            previous timestep.
        decoder_hidden_state : torch.LongTensor, optional (not needed if no attention)
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)
            Masks on encoder outputs. Needed only if using attention.
        """
        # input_indices : (batch_size/beam_size,)
        # (batch_size, target_embedding_dim)
        # input_indices = Variable(input_indices)
        # decoder_hidden_state = Variable(decoder_hidden_state)
        # encoder_outputs = Variable(encoder_outputs)
        # encoder_outputs_mask = Variable(encoder_outputs_mask)

        embedded_input = self._target_embedder(input_indices)
        if self._attention_function:
            # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
            # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
            # complain.
            encoder_outputs_mask = encoder_outputs_mask.float()
            # (batch_size, input_sequence_length)
            # print("decoder_hidden_state type: " + str(decoder_hidden_state))
            # print("encoder_outputs type: " + str(encoder_outputs))
            # print("encoder_outputs_mask: " + str(encoder_outputs_mask))

            # print("decoder_hidden_state size(): " + str(decoder_hidden_state.size()))
            # print("encoder_outputs size(): " + str(encoder_outputs.size()))
            # print("encoder_outputs_mask size(): " + str(encoder_outputs_mask.size()))

            # decoder_hidden_state = Variable(decoder_hidden_state)
            encoder_outputs = Variable(encoder_outputs)
            # encoder_outputs_mask = Variable(encoder_outputs_mask)

            input_weights = self._decoder_attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)
            # (batch_size, encoder_output_dim)
            attended_input = weighted_sum(encoder_outputs, input_weights)
            # print("attended_input size: " + str(attended_input))
            # print("embedded_input size: " + str(embedded_input))
            embedded_input = embedded_input.squeeze(1)
            # (batch_size, encoder_output_dim + target_embedding_dim)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleSeq2Seq':
        source_embedder_params = params.pop("source_embedder")
        source_embedder = TextFieldEmbedder.from_params(vocab, source_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        mmi = params.pop("mmi")
        beam_width = params.pop("beam_width")
        max_decoding_steps = params.pop("max_decoding_steps")
        target_namespace = params.pop("target_namespace", "tokens")
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        scheduled_sampling_ratio = params.pop_float("scheduled_sampling_ratio", 0.0)
        return cls(vocab,
                   source_embedder=source_embedder,
                   encoder=encoder,
                   mmi=mmi,
                   beam_width=beam_width,
                   max_decoding_steps=max_decoding_steps,
                   target_namespace=target_namespace,
                   attention_function=attention_function,
                   scheduled_sampling_ratio=scheduled_sampling_ratio)
