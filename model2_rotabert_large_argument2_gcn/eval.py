import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer,DistilBertTokenizer,CamembertTokenizer,RobertaTokenizer
import numpy as np
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,recall_score,f1_score
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
import torch.nn as nn
import time
import datetime
import random
import  json
from model import Bert_Model

MAX_LEN = 140



def labeltoOneHot(label):
    if label == "n":
        return [0, 1]
    else:  # indices == "y"
        return [1, 0]


def get_data(file):
    sentences = []
    labels = []
    graphs=[]
    with open(file, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = json.loads(line)
            sentence_ = line["sentence"]
            head_ = line["head"]
            relation = line["relation"]
            tail = line["tail"]
            label = line["label"]
            triple_ = head_ + " " + relation + " " + tail
            sentences.append(sentence_ + " " + triple_)
            labels.append(labeltoOneHot(label))
            graphs.append(line["graph"])

    return sentences, labels,graphs


def get_input_and_mask(sentences):
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    # Print sentence 0, now as a list of IDs.

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return input_ids, attention_masks

entities=[]
with open("data/parising_entity_relation_dict.txt", "r") as fr:
    for line in fr.readlines():
        entities.append(line.strip())

entity_idxs = {entities[i]: i for i in range(len(entities))}
def get_points(KG,Max_length_entity):
    nodes_all=[]
    after_edge_indices_all=[]
    edge_type_all=[]
    max_edge_index_e=[]
    for knowledge in KG:
        ht_r={}
        nodes=[]
        knowledge=knowledge.split(',')
        # print(knowledge)
        for triples in knowledge:
            # eraser|RelatedTo-RelatedTo|hand|0.3662654161453247
            triples = triples.split("|")
            if len(triples)==3:
                if triples[0] in entity_idxs:
                    head = entity_idxs[triples[0]]
                    nodes.append(head)
                else:
                    head=0
                    nodes.append(head)

                if triples[1] in entity_idxs:
                    relation1 = entity_idxs[triples[1]]
                else:
                    relation1 = 0

                if triples[2] in entity_idxs:
                    tail1 = entity_idxs[triples[2]]
                    nodes.append(tail1)
                else:
                    tail1 = 0
                    nodes.append(tail1)


                ht_r[str(head)+"|"+str(tail1)]=str(relation1)





        if len(nodes) < Max_length_entity:
            entity_nodes_id = nodes + [0] * (Max_length_entity - len(nodes))
        else:
            flags = random.sample(range(0, len(nodes)), Max_length_entity)
            entity_nodes_id = np.array(nodes)[flags]


        nodes = [x for x in entity_nodes_id]

        adjacency_matrix=[]
        # all_relations=[]
        for i in nodes:
            sub_adj=[]
            for j in nodes:
                if str(i)+"|"+str(j) in ht_r:
                    # all_relations.append(ht_r[str(i)+"|"+str(j)])
                    sub_adj.append(1)
                else:
                    sub_adj.append(0)

            adjacency_matrix.append(sub_adj)



        edge_indices = torch.tensor(adjacency_matrix).nonzero().t().contiguous()

        """
        tensor([[17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 26, 26, 26, 30, 35, 35, 35,
         35, 40, 40, 40, 40, 56, 56, 56, 56, 56, 60, 60, 60, 60, 60, 65, 65, 72,
         72, 72, 82, 82, 85, 85, 85, 92, 92, 92, 92, 92, 99, 99],
        [ 9, 14, 17, 36, 42, 48, 60, 35, 46, 60, 92, 56, 75, 84, 24, 18, 35, 48,
         75, 12, 24, 40, 98, 26, 56, 65, 72, 75, 12, 16, 17, 18, 62, 56, 65, 56,
         72, 85, 39, 82,  7, 72, 85, 12, 18, 49, 69, 92, 12, 63]])
        """

        after_edge_indices=[]

        flag = []
        edge_type = []
        start_sub_edge_indice=[]
        end_sub_edge_indice=[]
        for i in range(edge_indices.size(1)):
            start_points = edge_indices[0][i]
            end_points = edge_indices[1][i]
            start_sub_edge_indice.append(int(start_points))
            end_sub_edge_indice.append(int(end_points))

            two_ = str(start_points) + "|" + str(end_points)
            two_reverse = str(end_points) + "|" + str(start_points)
            flag.append(two_)
            if i == 0:
                edge_type.append(1)
            else:
                if two_ not in flag or two_reverse not in flag:
                    edge_type.append(1)
                if two_reverse in flag:
                    edge_type.append(0)


        after_edge_indices.append(start_sub_edge_indice)
        after_edge_indices.append(end_sub_edge_indice)

        nodes_all.append(nodes)
        after_edge_indices_all.append(after_edge_indices)
        edge_type_all.append(edge_type)


        max_edge_index_e.append(len(edge_type))


    Max_e_edge = max(max_edge_index_e)
    end_e_edge_indices_batch = []
    for tupple_e in after_edge_indices_all:
        new_tuple = []
        if len(tupple_e[0]) < Max_e_edge:
            new_tuple.append(tupple_e[0] + [0] * (Max_e_edge - len(tupple_e[0])))
            new_tuple.append(tupple_e[1] + [0] * (Max_e_edge - len(tupple_e[1])))
        else:
            new_tuple.append(tupple_e[0])
            new_tuple.append(tupple_e[1])
        #     # print("mmmm",len(new_tuple))
        end_e_edge_indices_batch.append(new_tuple)

    end_e_edge_type_batch = []
    for type_e in edge_type_all:
        if len(type_e) < Max_e_edge:
            end_e_edge_type_batch.append(type_e + [0] * (Max_e_edge - len(type_e)))
        else:
            end_e_edge_type_batch.append(type_e)



    return nodes_all,end_e_edge_indices_batch,end_e_edge_type_batch

def get_input_id():
    kg_node_length=104
    train_sentences, train_labels,graphs_train = get_data("data/sentence_triple_graph_train.json")
    e_nodes_train, e_edge_indices_train, e_edge_type_train = get_points(graphs_train, kg_node_length)
    # print(e_nodes)
    # print(e_edge_indices)
    # print(e_edge_type)

    train_input_ids, train_attention_masks = get_input_and_mask(train_sentences)
    #
    test_sentences, test_labels,graphs_test = get_data("data/sentence_triple_graph_test.json")
    test_input_ids, test_attention_masks = get_input_and_mask(test_sentences)
    e_nodes_test, e_edge_indices_test, e_edge_type_test = get_points(graphs_test, kg_node_length)
    train_inputs = torch.tensor(train_input_ids)
    test_inputs = torch.tensor(test_input_ids)
    train_labels = torch.FloatTensor(train_labels)
    test_labels = torch.FloatTensor(test_labels)
    train_masks = torch.tensor(train_attention_masks)
    test_masks = torch.tensor(test_attention_masks)


    e_nodes_train=torch.tensor(e_nodes_train)
    e_edge_indices_train=torch.tensor(e_edge_indices_train)
    e_edge_type_train=torch.tensor(e_edge_type_train)
    e_nodes_test=torch.tensor(e_nodes_test)
    e_edge_indices_test=torch.tensor(e_edge_indices_test)
    e_edge_type_test=torch.tensor(e_edge_type_test)







    #
    batch_size = 32
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels,e_nodes_train,e_edge_indices_train,e_edge_type_train)
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our validation set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels,e_nodes_test,e_edge_indices_test,e_edge_type_test)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader


def flat_PRF1(preds, labels):
    pred_flat = preds
    labels_flat = labels
    P=accuracy_score(pred_flat, labels_flat)
    R=recall_score(pred_flat, labels_flat)
    F1=f1_score(pred_flat, labels_flat)
    return P,R,F1
#

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))




def TE(model, validation_dataloader):
    model.eval()
    # Tracking variables
    test_loss, test_accuracy ,test_f1,test_recall= 0, 0,0,0
    nb_test_steps, nb_test_examples = 0, 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.cuda() for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels,e_nodes_train,e_edge_indices_train,e_edge_type_train= batch


        # speeding up validation
        with torch.no_grad():
            outputs, _ = model(input=b_input_ids,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               e_nodes_batch=e_nodes_train,
                               e_edge_indices_batch=e_edge_indices_train,
                               e_edge_type_batch=e_edge_type_train

                               )

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs
        logits = torch.argmax(logits, dim=1)
        b_labels = torch.argmax(b_labels, dim=1)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        P, R, F1=flat_PRF1(logits, label_ids)

        # Accumulate the total accuracy.
        test_accuracy += P
        test_f1+=F1
        test_recall+=R
        # Track the number of batches
        nb_test_steps += 1
    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(test_accuracy / nb_test_steps))
    print("  Recall: {0:.2f}".format(test_recall / nb_test_steps))
    print("  F1: {0:.2f}".format(test_f1 / nb_test_steps))
    # print("  test took: {:}".format(format_time(time.time() - t0)))

    global_P = test_accuracy / nb_test_steps
    global_R = test_recall / nb_test_steps
    global_F1 = test_f1 / nb_test_steps

    return global_P,global_R,global_F1




if __name__ == "__main__":
    patience = 5
    classes = 2

    train_dataloader, test_dataloader = get_input_id()

    model = Bert_Model(classes)

    model.cuda()
    best_model = model.state_dict()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # args.adam_epsilon  - default is 1e-8. )

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Create the learning rate scheduler.
    fname = 'checkpoints/best_score_model.pt'
    model.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
    print("-------the test set performance------")
    global_P,global_R,global_F1 = TE(model, test_dataloader)
    print("-------the train set performance------")
    train_global_P, train_global_R, train_global_F1 = TE(model, train_dataloader)
