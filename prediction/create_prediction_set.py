import json
import random


data_path = "../corpus_max_350/corpus_0_test.tsv"

lines = open(data_path).read().splitlines()

# take out just the questions
questions = [line.split("\t")[0] for line in lines]

# filter questions
newQ = [question for question in questions
        if len(question.split()) >= 20 and len(question.split()) < 80]

random.shuffle(newQ)

# take part of the set
newQ = newQ[:40]

# dump the data
with open("./prediction.jsonl", 'w') as output:
    for question in newQ:
        output.write(json.dumps({"question": question}) + "\n")
