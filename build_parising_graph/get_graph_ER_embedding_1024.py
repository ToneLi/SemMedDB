from sentence_transformers import SentenceTransformer, models
from torch import nn
import numpy as np

word_embedding_model = models.Transformer('roberta-large', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=1024, activation_function=nn.Tanh())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

"""
Adjacent to:   this entity has an adjacency with the entities before and after
composition: composition relation for child entity or father entity


"""
E=[]
i=0
with open("parising_entity_relation_dict.txt","r",encoding="utf-8")  as fr:
    for line in fr.readlines():
        i=i+1
        print(i)
        words_=line.strip()
        sentence_embeddings = model.encode([words_])
        E.append(sentence_embeddings[0])


E=np.array(E)
np.save("Graph_ER_embedding__1024.npy",E)
