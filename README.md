## SemMedDB
This task is about to extract a triple from a sentence, so I regard it as a classifcation task. To solve this task, I proposed three method Sequence vs. Data augmentation vs. Graph.

## Environment

```
python3.6.5
torch 1.4.0+cu100
TextAttack 0.2.5
```
## Preprogress
You can skip this part, this step tells us:

how to change the data to json file: 0_progress_file_extraction.py, 

how to plot the sentence length chart: 1_analysis_data.py, 

how to split the data into train and test set: 2_split_train_test.py, 

how to plot the loss 3_plot_loss.py. 

The final data is data_progress/final_data/sentence_triple_train.jsonl and data_progress/final_data/sentence_triple_test.jsonl
