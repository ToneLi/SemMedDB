from transformers import BertModel,DistilBertModel,CamembertModel,RobertaModel
import  torch.nn as nn
import torch
from rgcn_custom import RGCNConv
import numpy as np
from graph_utils import get_pytorch_graph
import torch
import torch.nn.functional as F

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states

class T5GNNAdapt(nn.Module):
    def __init__(self):
        super().__init__()
        d_model=1024
        d_ff=286
        dropout_rate=0.5
        self.conv = RGCNConv(d_model, d_ff, num_relations=2, root_weight=True)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

        self.layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_gnn = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, edge_indices, edge_type):
        # print("hidden_states",hidden_states.size())
        # print("hidden_states",hidden_states)
        hidden_states = hidden_states.cuda()
        norm_x = self.layer_norm(hidden_states)
        # print(norm_x)
        # print(edge_indices)
        # print(edge_type)
        graph_batch = get_pytorch_graph(norm_x, edge_indices, edge_type)
        # print("graph_batch", graph_batch)
        # print("graph_batch.x",graph_batch.x)
        # print("graph_batch.edge_index", graph_batch.edge_index)
        # print("graph_batch.y", graph_batch.y)
        y = F.elu(self.conv(graph_batch.x, graph_batch.edge_index, edge_type=graph_batch.y))
        # print("kkkkkkkkk",y.size())
        y = self.dropout_gnn(y)
        y = self.wo(y)
        y = y.view_as(hidden_states)

        layer_output = hidden_states + self.dropout(y)
        return layer_output

class Bert_Model(nn.Module):
   def __init__(self, classes):
       super(Bert_Model, self).__init__()
       self.bert = RobertaModel.from_pretrained('roberta-large')
       # self.out = nn.Linear(self.bert.config.hidden_size, classes)
       self.loss = torch.nn.BCELoss(reduction='sum')
       self.dropout = nn.Dropout(0.1)
       self.label_smoothing=0
       # relu activation function
       self.relu = nn.ReLU()

       # dense layer 1
       # self.fc1 = nn.Linear(768, 512)

       W1 = torch.ones(400, 768)
       W1 = nn.init.uniform_(W1)
       self.W1 = nn.Parameter(W1)
       self.fc1 = nn.Linear(1024, 512)
       # dense layer 2 (Output layer)
       self.fc2 = nn.Linear(512, 2)

       # softmax activation function
       self.softmax =torch.nn.Softmax(dim=1)

       self.Entity_embedding = np.load("data/Graph_ER_embedding.npy")
       self.E_embedding_knowledge = torch.nn.Embedding( self.Entity_embedding.shape[0],  self.Entity_embedding.shape[1]).cuda()
       self.E_embedding_knowledge.weight.data.copy_(torch.from_numpy( self.Entity_embedding))
       self.E_embedding_knowledge.weight.requires_grad = True

       self.adapter_GNN = T5GNNAdapt()

   def applyNonLinear(self, question_embedding):
       x = self.fc1(question_embedding)
       x = self.relu(x)
       x = self.dropout(x)
       # output layer
       x = self.fc2(x)
       return x

   def get_triple_representation(self,question,head_entity):
   

       Topic_entity=torch.matmul(head_entity, self.W1)

       pi = 3.14159265358979323846
       re_head, im_head = torch.chunk(Topic_entity, 2, dim=1)
       re_relation, im_relation = torch.chunk(question, 2, dim=1)

       re_head=torch.cos(re_head)/pi
       im_head=torch.sin(im_head)/pi

       re_relation = torch.cos(re_relation)/pi
       im_relation = torch.sin(im_relation)/pi
       # print("re_relation",re_relation.size())
       # print("im_relation",im_relation.size())
       #        # print("re_head",re_head.size())
       #        # print("im_head",im_head.size())
       re_tail = re_relation * re_head - im_relation * im_head
       im_tail = im_head * re_relation + re_head * im_relation

       # print("re_score", re_score.size())
       # print("im_score", im_score.size())

       tail = torch.cat([re_tail, im_tail], dim=1)
       # score = score.norm(dim=0)


       triple_repre=Topic_entity+tail+question

       return  triple_repre
   def forward(self, input,attention_mask,labels,
                            e_nodes_batch,
                                e_edge_indices_batch,
                                e_edge_type_batch):
       fw=open("results.txt","w")

       question_embedding= self.bert(input, attention_mask = attention_mask)
       question_embedding=torch.mean(question_embedding["last_hidden_state"],dim=1)


       # graph_nodes_embedding =  self.E_embedding_knowledge(e_nodes_batch)
       # nodes_knowledge = self.adapter_GNN(graph_nodes_embedding, e_edge_indices_batch, e_edge_type_batch)

       # e_edge_indices_batch=e_edge_indices_batch.cuda()
       # e_edge_type_batch=e_edge_type_batch.cuda()
       graph_nodes_embedding = self.E_embedding_knowledge(e_nodes_batch)
       nodes_knowledge = self.adapter_GNN(graph_nodes_embedding, e_edge_indices_batch, e_edge_type_batch)
       # print("kkkkkkkkkkkkkkk",nodes_knowledge.size())

       question_embedding=question_embedding+0.02*torch.mean(nodes_knowledge,dim=1)

       x = self.applyNonLinear(question_embedding)
       question_embedding= self.softmax(x)
       actual_r = labels

       loss = self.loss(question_embedding, actual_r)

       return question_embedding,loss

