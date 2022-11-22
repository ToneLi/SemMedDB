
"""

the speed of dic is fast, so I need to change  the file to jsonl file
and to remove the long space  by using the replace in python
"""

import pandas as pd
import json
fw = open("sentence_triple.jsonl", "w")
s=[]
df = pd.read_csv("data/sentence_triple_yes_no.csv")

id=df["PREDICATION_ID"].values
head=df["SUBJECT_TEXT"].values
relation=df["PREDICATE"].values
tail=df["OBJECT_TEXT"].values
sentence=df["SENTENCE"].values
label=df["LABEL"].values

i=0
for i  in range(len(label)):
    DIC = {}
    DIC["ID"] = id[i]
    DIC["head"] = head[i].replace("       "," ")
    DIC["relation"] =  relation[i]
    DIC["tail"] = tail[i]
    DIC["sentence"] =  sentence[i].replace("       "," ")
    DIC["label"] =  label[i]
    fw.writelines(json.dumps(DIC) + "\n")
    fw.flush()
