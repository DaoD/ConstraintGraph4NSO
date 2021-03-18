import torch
from torch.utils.data import TensorDataset


class Dataset(TensorDataset):
    def __init__(self, X_input_ids, X_attention_mask, X_token_type_ids, X_adj1, X_adj2, X_adj3, X_story_len, y_labels=None):
        super(Dataset, self).__init__()
        X_input_ids = torch.LongTensor(X_input_ids)
        X_attention_mask = torch.LongTensor(X_attention_mask)
        X_token_type_ids = torch.LongTensor(X_token_type_ids)
        X_adj1 = torch.FloatTensor(X_adj1)
        X_adj2 = torch.FloatTensor(X_adj2)
        X_adj3 = torch.FloatTensor(X_adj3)
        X_story_len = torch.LongTensor(X_story_len)

        if y_labels is not None:
            y_labels = torch.LongTensor(y_labels)
            self.tensors = [X_input_ids, X_attention_mask, X_token_type_ids, X_adj1, X_adj2, X_adj3, X_story_len, y_labels]
        else:
            self.tensors = [X_input_ids, X_attention_mask, X_token_type_ids, X_adj1, X_adj2, X_adj3, X_story_len]

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])