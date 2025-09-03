import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self , d_model: int , vocab_size : int )->None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model)

    def forward(self , x):
        # (batch , seq_len) --> (batch , seq_len , d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositonalEncoding(nn.Module):

    def __init__(self , d_model : int , seq_len : int , dropout : float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of (seq_len ,  d_model)
        pe = torch.zeros(seq_len , d_model)

        # create a vector of shape seq_len
        # (seq_len , 1)
        position = torch.arrange(0 , seq_len , dtype = torch.float).unsqueeze(-1)

        # create a vector of shape (d_model)

        # (d_model / 2 )
        div_term = torch.exp(torch.arrange(0 , d_model ,2).float() * (-math.log(10000.0)/d_model))

        # apply sine to even indices
        pe[:, 0::2]  = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))

        #apply cosine to odd indices
        pe[:, 1:2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

        pe =  pe.unsqueeze(0) # (1,seq_len , d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe',pe)

    def forward(self ,x):
        x = x  + (self.pe[: , :x.shape[1], : ]).requires_grad(False) # (batch , seq_len , d_model)
        return self.dropout(x)

        
        