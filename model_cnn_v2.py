from typing import Dict

import torch
from torch.nn import Embedding
from torch.autograd import Variable
import torch.nn.functional as F
import math
import random
import numpy as np


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        batch_size: int,
    ) -> None:
        super(SeqClassifier, self).__init__()

        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.batch_size = batch_size
        self.dropout = torch.nn.Dropout(p=0.25)
        self.embedding_size = embeddings.shape[1]
        # self.dropout = dropout

        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=35, kernel_size = self.kernel_1)
        self.conv2 = torch.nn.Conv1d(in_channels=1, out_channels=35, kernel_size = self.kernel_2)
        self.conv3 = torch.nn.Conv1d(in_channels=1, out_channels=35, kernel_size = self.kernel_3)
        self.conv4 = torch.nn.Conv1d(in_channels=1, out_channels=35, kernel_size = self.kernel_4)

        self.maxpool1 = torch.nn.MaxPool1d(kernel_size = self.kernel_1)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size = self.kernel_2)
        self.maxpool3 = torch.nn.MaxPool1d(kernel_size = self.kernel_3)
        self.maxpool4 = torch.nn.MaxPool1d(kernel_size = self.kernel_4)

        self.lstm = torch.nn.LSTM(11372, hidden_size, batch_first=True, bidirectional = self.bidirectional)

        if bidirectional == True:
            self.linear = torch.nn.Linear(self.hidden_size*2, self.num_class)
        elif bidirectional == False:
            self.linear = torch.nn.Linear(self.hidden_size, self.num_class) 
        # TODO: model architecture

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self.embed(batch)
        (batch_size, seq_len, embed_shape) = x.shape
        x = torch.reshape(x, (batch_size, 1, -1))

        x1 = F.relu(self.conv1(x))
        x1 = self.maxpool1(x1)
        x1 = self.dropout(x1)

        x2 = F.relu(self.conv2(x))
        x2 = self.maxpool2(x2)
        x2 = self.dropout(x2)

        x3 = F.relu(self.conv3(x))
        x3 = self.maxpool3(x3)
        x3 = self.dropout(x3)

        # x4 = F.relu(self.conv4(x))
        # x4 = self.maxpool4(x4)
        # x4 = self.dropout(x4)


        union = torch.cat((x1, x2, x3), 2)
        # union = union.reshape(union.size(0), -1)
        # print(union.shape)

        union = self.dropout(union)

        out,_ = self.lstm(union)
        output = self.linear(out)
        return output

