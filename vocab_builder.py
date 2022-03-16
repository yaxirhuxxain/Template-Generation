# -*- coding: utf-8 -*-


import os
import json
import pickle
from collections import Counter

javaDir = './data/java'
javaProjects = [name for name in os.listdir(javaDir)]

words = Counter()

for i, name in enumerate(javaProjects):
    print(f"*** Processing {i + 1}/{len(javaProjects)}: {name}")
    data = open(os.path.join(javaDir, name), "r", encoding="utf-8").read()

    for file in data.split('\n'):
        code = list(filter(None, file.split()))
        words.update(code)

n = 500
words_to_keep = [i[0] for i in words.most_common(n)]
words_to_keep_total = sum(i[1] for i in words.most_common(n))

total = sum(words.values())

print("Total # of words: {}".format(len(words)))
print(f"Top {n} words covers: {100 * words_to_keep_total / total:.2f}% of the dataset")

print("Top 10 common words:")
for v, i in words.most_common(10):
    print(v, i)

# dump vocab to file
print("dump top-500-words count to pickle file")
with open("top-500-words.pkl", "wb") as fout:
    pickle.dump(words_to_keep, fout)

print("building and dumping word_idx.json")
# building word to idx encoder
word_idx = {"<pad>": 0, "<idf>": 1}
counter = len(word_idx)
for word in words_to_keep:
    word_idx[word] = counter
    counter += 1

with open("word_idx.json", "w") as outfile:
    json.dump(word_idx, outfile, indent=4)
