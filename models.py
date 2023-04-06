import torch
from torch import nn
import torch.nn.functional as F

# FEMNIST / CIFAR
class CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNN, self).__init__()
        self.in_dim = in_dim
        in_c, in_h, in_w = in_dim
        self.conv1 = nn.Conv2d(in_c, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout2d(0.0)
        linear_in_dim = 64*(in_h//2 - 2)*(in_w//2 - 2)
        self.fc1 = nn.Linear(linear_in_dim, 1024)
        self.fc2 = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = x.view(-1, *self.in_dim)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        n_hidden,
        num_embeddings,
        embedding_dim,
        max_seq_len,
        dropout_rate,
        full_seq=False,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_embeddings = num_embeddings
        self.full_seq = full_seq

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.fc = nn.Linear(self.n_hidden, self.num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        seq_lens = torch.sum(x != (self.num_embeddings - 1), 1) - 1
        x = self.embedding(x)  # [B, S] -> [B, S, E]
        out, _ = self.lstm(x)  # [B, S, E] -> [B, S, H]
        # if self.full_seq:
        #     print(out.shape)
        #     out = out[torch.arange(out.size(0)), seq_lens]
        out = self.fc(self.dropout(out))  # [B, S, H] -> # [B, S, C]
        return out



# Synthetic
class LogisticRegression(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = nn.Linear(input_dim, output_dim)
         self.dropout = nn.Dropout(p=0.0) # dummy
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs
