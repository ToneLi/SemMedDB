import  json
y=0
n=0
with open("sentence_triple_train_all_argument.jsonl","r") as fr:
    for line in fr.readlines():
        line=json.loads(line)
        labels=line["label"]
        if labels=="y":
            y=y+1
        if labels=="n":
            n=n+1

print(y)
print(n)

"""
before:
1284
1116


"""
