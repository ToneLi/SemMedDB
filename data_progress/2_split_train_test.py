"""
k-fold cross-validation is  the best way to valid the model,
 in this work I used  sklearn to split the train and test
 the k-fold cross validation excample:
 from sklearn.model_selection import KFold
import numpy as np
X = np.arange(24).reshape(12,2)
y = np.random.choice([1,2],12,p=[0.4,0.6])
kf = KFold(n_splits=5,shuffle=False)
for train_index , test_index in kf.split(X):
    print('train_index:%s , test_index: %s ' %(train_index,test_index))

In this work, I regard this task as a 2 label classification
"""


import json
from sklearn.model_selection import train_test_split
i=0
all_=[]
labels=[]
with open("final_data/sentence_triple.jsonl","r",encoding="utf-8") as fr:
    for line in fr.readlines():
        i=i+1
        line=json.loads(line)
        all_.append(line)
        labels.append(line["label"])

train_X,test_X,train_y,test_y = train_test_split(all_,labels,test_size=0.2,random_state=5)
fw_train = open("sentence_triple_train.jsonl", "w",encoding="utf-8")
fw_test = open("sentence_triple_test.jsonl", "w",encoding="utf-8")

for unit in train_X:
    fw_train.writelines(json.dumps(unit) + "\n")
    fw_train.flush()

for uni in test_X:
    fw_test.writelines(json.dumps(uni) + "\n")
    fw_test.flush()