### SemMedDB
This task is about to extract a triple from a sentence, so I regard it as a classifcation task. To solve this task, I proposed three method Sequence vs. Data augmentation vs. Graph.

### Environment

```
python3.6.5
torch 1.4.0+cu100
TextAttack 0.2.5
```
### Preprogress
You can skip this part, this step tells us:

how to change the data to json file: **0_progress_file_extraction.py**

how to plot the sentence length chart: **1_analysis_data.py**

how to split the data into train and test set: **2_split_train_test.py** 

how to plot the loss **3_plot_loss.py**

The final data is **data_progress/final_data/sentence_triple_train.jsonl** and **data_progress/final_data/sentence_triple_test.jsonl**

### Data argumentation
Three Strategies is in the file data_argument. There is an excample for the wordnet argument method

```
from textattack.augmentation import WordNetAugmenter

text = "In both the oxidase activity as well as the MI complex formation phenobarbital induced cytochrome P-450 is involved"

wordnet_aug = WordNetAugmenter()

wordnet_aug.augment(text)
```
### Build the dependency parse graph
The progress is 
