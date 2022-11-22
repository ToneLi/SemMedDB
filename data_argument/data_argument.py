# encoding: utf-8
# pip install textattack.0.2.5
# python 3.6
import json
"""
refer: link
https://www.freecodecamp.org/news/how-to-perform-data-augmentation-in-nlp-projects/
"""
from textattack.augmentation import EasyDataAugmenter,WordNetAugmenter,CharSwapAugmenter
deletion_aug = EasyDataAugmenter()
wordnet_aug = WordNetAugmenter()
charswap_aug = CharSwapAugmenter()
# deletion_aug.augment(text)
def get_argument_sentence(text):
    # return wordnet_aug.augment(text)# +wordnet_aug.augment(text)+charswap_aug.augment(text)
    return deletion_aug.augment(text)+wordnet_aug.augment(text)+charswap_aug.augment(text)

fw= open("data/sentence_triple_train_all_argument.jsonl","w",encoding="utf=8")
i=0
with open("data/sentence_triple_train.jsonl","r",encoding="utf=8") as fr:
    for line in fr.readlines():
        i=i+1
        print(i)
        sen_=json.loads(line)
        fw.writelines(json.dumps(sen_) + "\n")
        sentence_=get_argument_sentence(sen_["sentence"])
        # print(sentence_)
        for s in sentence_:
            if len(s)==0:
                print("------")
            sen_["sentence"]=s
            fw.writelines(json.dumps(sen_) + "\n")

        fw.flush()
        # break

