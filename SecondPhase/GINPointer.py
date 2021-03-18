import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from NeuralNetwork import NeuralNetwork
from Model import GraphIsomorphismNetwork
from transformers import BertModel, AlbertModel, RobertaModel


class GIN4Ordering(NeuralNetwork):
    def __init__(self, args, logger=None):
        self.args = args
        super(GIN4Ordering, self).__init__()
        self.hidden_size = 512
        self.bert_size = 768
        self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self.bert_encoder = AlbertModel.from_pretrained("albert-base-v2")
        # self.bert_encoder = RobertaModel.from_pretrained("roberta-base")
        self.gin1 = GraphIsomorphismNetwork(node_feature_dim=self.bert_size, hidden_dim=[self.hidden_size, self.hidden_size], output_dim=self.hidden_size, activation="relu", epsilon=0, batch_norm=True)
        self.gin2 = GraphIsomorphismNetwork(node_feature_dim=self.bert_size, hidden_dim=[self.hidden_size, self.hidden_size, self.hidden_size], output_dim=self.hidden_size, activation="relu", epsilon=0, batch_norm=True)
        self.gin3 = GraphIsomorphismNetwork(node_feature_dim=self.bert_size, hidden_dim=[self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size], output_dim=self.hidden_size, activation="relu", epsilon=0, batch_norm=True)
        self.FFFN1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.FFFN2 = nn.Linear(self.hidden_size, 1)
        self.logger = logger
        self._inf = nn.Parameter(torch.FloatTensor([-1e9]), requires_grad=False)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        init.xavier_normal_(self.FFFN1.weight)
        init.xavier_normal_(self.FFFN2.weight)

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

    def forward(self, input_ids, attention_mask, token_type_ids, adj_matrix_1, adj_matrix_2, adj_matrix_3, story_len, is_train=False):
        """
        Args:
            input_ids ([Tensor]):       [batch, num_sentences, sentence_len]
            attention_mask ([Tensor]):  [batch, num_sentences, sentence_len]
            token_type_ids ([Tensor]):  [batch, num_sentences, sentence_len]
            adj_matrix ([Tensor]):      [batch, num_sentences, num_sentences]
            story_len([Tensor]): [batch]
        """
        batch_size = input_ids.size(0)
        num_sentences = input_ids.size(1)
        node_reps = []
        for i in range(num_sentences):
            bert_inputs = {'input_ids': input_ids[:, i, :], 'attention_mask': attention_mask[:, i, :], 'token_type_ids': token_type_ids[:, i, :]}
            sent_rep = self.dropout(self.bert_encoder(**bert_inputs)[1])
            node_reps.append(sent_rep)

        node_reps = torch.stack(node_reps, dim=1)  # [batch, num_sentences, hidden_size]
        out_bert_rep = node_reps
        node_reps1 = self.gin1(node_reps, adj_matrix_1)  # [batch, hidden_size], [batch, num_sentences, hidden_size]
        node_reps2 = self.gin2(node_reps, adj_matrix_2)  # [batch, hidden_size], [batch, num_sentences, hidden_size]
        node_reps3 = self.gin3(node_reps, adj_matrix_3)  # [batch, hidden_size], [batch, num_sentences, hidden_size]
        node_scores = self.FFFN1(torch.cat([node_reps1, node_reps2, node_reps3], dim=-1)) # [batch, num_sentences]
        node_scores = self.FFFN2(self.relu(node_scores)).squeeze(-1)
        story_len = story_len.unsqueeze(1)
        mask = torch.arange(num_sentences).cuda().unsqueeze(0).expand(batch_size, num_sentences) >= story_len  # [batch, num_sentences]
        self.init_inf(mask.size())
        node_scores[mask] = self.inf[mask]
        y_indices = node_scores.argsort(dim=-1, descending=True)
        return node_scores, y_indices