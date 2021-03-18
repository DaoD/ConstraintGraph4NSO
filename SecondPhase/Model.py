from collections import *
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from transformers import BertModel, AlbertModel

class GraphIsomorphismNetworkLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation, batch_norm):
        super(GraphIsomorphismNetworkLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = False

        if activation is not None:
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        x = self.linear(x)
        B, N, d = x.size()
        x = x.reshape(B * N, d)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = x.reshape(B, N, d)
        if self.activation:
            x = self.activation(x)
        return x


class GraphIsomorphismNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, activation, epsilon, batch_norm=True):
        super(GraphIsomorphismNetwork, self).__init__()
        self.gin_layers_dim = [node_feature_dim] + hidden_dim
        self.gin_layers_num = len(self.gin_layers_dim)
        self.epsilon = epsilon
        self.output_dim = output_dim
        self.batch_norm = batch_norm

        self.gin_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.gin_layers_dim[:-1], self.gin_layers_dim[1:])):
            self.gin_layers.append(GraphIsomorphismNetworkLayer(in_dim, out_dim, activation, batch_norm))

        self.fc_layers = nn.Linear(sum(self.gin_layers_dim), self.output_dim)

        init.xavier_normal_(self.fc_layers.weight)

    def represent(self, x, adjacency):
        h = []
        h.append(x)
        for layer_idx in range(self.gin_layers_num-1):
            x = (1 + self.epsilon) * x + torch.bmm(adjacency, x)
            x = self.gin_layers[layer_idx](x)
            # x_sum = torch.sum(x, dim=1)
            h.append(x)
        h = torch.cat(h, dim=-1)
        return h

    def forward(self, x, adjacency):
        h = self.represent(x, adjacency)
        h = self.fc_layers(h)
        return h

class GraphConvolutionNetworkLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation, batch_norm):
        super(GraphConvolutionNetworkLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = False

        if activation is not None:
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        return

    def forward(self, x):
        x = self.linear(x)
        B, N, d = x.size()
        x = x.reshape(B*N, d)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = x.reshape(B, N, d)
        if self.activation:
            x = self.activation(x)
        return x


class GraphConvolutionNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, activation, batch_norm=True):
        super(GraphConvolutionNetwork, self).__init__()
        self.gcn_layers_dim = [node_feature_dim] + hidden_dim
        self.gcn_layers_num = len(self.gcn_layers_dim)
        self.output_dim = output_dim
        self.batch_norm = batch_norm

        self.gcn_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.gcn_layers_dim[:-1], self.gcn_layers_dim[1:])):
            self.gcn_layers.append(GraphConvolutionNetworkLayer(in_dim, out_dim, activation, batch_norm))

        self.fc_layers = nn.Linear(self.gcn_layers_dim[-1], self.output_dim)
        return

    def represent(self, x, adjacency):
        for layer_idx in range(self.gcn_layers_num-1):
            x = x + torch.bmm(adjacency, x)
            x = self.gcn_layers[layer_idx](x)
            x_sum = torch.sum(x, dim=1)
            h = x_sum
        return h

    def forward(self, x, adjacency):
        x = self.represent(x, adjacency)
        x = self.fc_layers(x)
        return x


class BertClassification(nn.Module):
    def __init__(self):
        super(BertClassification, self).__init__()
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.output_mlp = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.ReLU(),
            nn.Linear(768, 2)
        )
        self.loss_func = nn.CrossEntropyLoss()

        init.xavier_normal_(self.output_mlp[0].weight)
        init.xavier_normal_(self.output_mlp[2].weight)

    def forward(self, sent_a_id, sent_a_attn, sent_a_type, sent_b_id, sent_b_attn, sent_b_type, y_label):
        bert_inputs = {'input_ids': sent_a_id, 'attention_mask': sent_a_attn, 'token_type_ids': sent_a_type}
        sent_a_rep = self.bert_encoder(**bert_inputs)[1]
        bert_inputs = {'input_ids': sent_b_id, 'attention_mask': sent_b_attn, 'token_type_ids': sent_b_type}
        sent_b_rep = self.bert_encoder(**bert_inputs)[1]
        features = torch.cat([sent_a_rep, sent_b_rep, torch.abs(sent_a_rep - sent_b_rep)], dim=-1)
        output = self.output_mlp(features)
        loss = self.loss_func(output, y_label)
        return output, loss

class BertClassification2(nn.Module):
    def __init__(self):
        super(BertClassification2, self).__init__()
        self.bert_encoder = AlbertModel.from_pretrained('albert-base-v2')
        self.bert_size = 768
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_size * 3, self.bert_size),
            nn.ReLU(),
            nn.Linear(self.bert_size, 2)
        )
        self.loss_func = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)
        init.xavier_normal_(self.classifier[0].weight)
        init.xavier_normal_(self.classifier[2].weight)

    def forward(self, sent_a_id, sent_a_attn, sent_a_type, sent_b_id, sent_b_attn, sent_b_type, y_label):
        bert_inputs = {'input_ids': sent_a_id, 'attention_mask': sent_a_attn, 'token_type_ids': sent_a_type}
        sent_a_rep = self.dropout(self.bert_encoder(**bert_inputs)[1])
        bert_inputs = {'input_ids': sent_b_id, 'attention_mask': sent_b_attn, 'token_type_ids': sent_b_type}
        sent_b_rep = self.dropout(self.bert_encoder(**bert_inputs)[1])
        sim = torch.cat([sent_a_rep, sent_b_rep, torch.abs(sent_a_rep - sent_b_rep)], dim=-1)
        output = self.classifier(sim)
        loss = self.loss_func(output, y_label)
        return output, loss

class TransformerBlock(nn.Module):
    def __init__(self, input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V, episilon=1e-8):
        """
        :param Q: (batch_size, max_r_words, embedding_dim)
        :param K: (batch_size, max_u_words, embedding_dim)
        :param V: (batch_size, max_u_words, embedding_dim)
        :return: output: (batch_size, max_r_words, embedding_dim)  same size as Q
        """
        dk = torch.Tensor([max(1.0, Q.size(-1))]).cuda()
        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon)
        # (batch_size, max_r_words, max_u_words)
        Q_K_score = F.softmax(Q_K, dim=-1)
        V_att = Q_K_score.bmm(V)
        if self.is_layer_norm:
            # (batch_size, max_r_words, embedding_dim)
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output