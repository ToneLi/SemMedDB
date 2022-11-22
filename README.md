### SemMedDB
This task is about to extract a triple from a sentence, so I regard it as a classifcation task. To solve this task, I proposed three method Sequence vs. Data augmentation vs. Graph. If you want to run the model, please skip into step4.

### Environment

```
python3.6.5
torch 1.4.0+cu100
TextAttack 0.2.5

```
### step1: Preprogress
You can skip this part, this step tells us:

how to change the data to json file: **0_progress_file_extraction.py**

how to plot the sentence length chart: **1_analysis_data.py**

how to split the data into train and test set: **2_split_train_test.py** 

how to plot the loss **3_plot_loss.py**

The final data is **data_progress/final_data/sentence_triple_train.jsonl** and **data_progress/final_data/sentence_triple_test.jsonl**

### step 2 Data argumentation
Three Strategies is in the file data_argument. There is an excample for the wordnet argument method

```
from textattack.augmentation import WordNetAugmenter

text = "In both the oxidase activity as well as the MI complex formation phenobarbital induced cytochrome P-450 is involved"

wordnet_aug = WordNetAugmenter()

wordnet_aug.augment(text)
```
### step 3 Build the dependency parse graph
The progress about dependency parse graph is in the file  build_parising_graph.
code demo

```
import json
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
def add_parsing_graph(file,stype):
    ER=[]
    # fw=open("sentence_triple_graph_%s.json"%stype,"w",encoding="utf-8")
    entity_length=[]
    with open(file,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            E=[]
            line=json.loads(line)
            sentence=line["sentence"].replace("[","").replace("]","")
            doc = nlp(sentence)
            triples=""
            for token in doc:
                head=token.head.text
                ER.append(head)
                E.append(head)
                relation=token.dep_
                ER.append(relation)
                tail=token.text
                E.append(tail)
                ER.append(tail)
                triple=head+"|"+relation+"|"+tail
                triples=triples+triple+","
            triples=triples.strip(",")
            line["graph"]=triples
            entity_length.append(len(set(E)))
    return list(set(ER)),max(entity_length)
```
### Step 4 To run the model
Note: For all model1, the data is same, it is in the file model1_bert_base. In the model model2_rotabert_large_argument2_gcn, the file Graph_ER_embedding.npy is in [Google Cloud](https://drive.google.com/file/d/1pkjLOq3lxReAqHh9tbw9nIEcaFkza0wp/view?usp=share_link)

For each model please run:
```
Train: CUDA_VISIBLE_DEVICES=3 python train.py
Test:  CUDA_VISIBLE_DEVICES=3 python eval.py
Please use the default command
```
