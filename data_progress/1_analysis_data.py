import json

import matplotlib.pyplot as plt

length_=[]
i=0
with open("final_data/sentence_triple.jsonl","r",encoding="utf-8") as fr:
    for line in fr.readlines():
        i=i+1
        line=json.loads(line)
        sentence_=line["sentence"]
        length_.append(len(sentence_.split(" ")))
        # break
x=list(range(i))
# print(length_)
plt.xlabel('sentence id')
plt.ylabel('lengths')
plt.title("the length of sentences")
plt.plot(x, length_)
plt.show()