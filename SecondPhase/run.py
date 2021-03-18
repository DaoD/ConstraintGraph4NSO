import time
import argparse
import pickle
import random
import numpy as np
import torch
import logging
from GINPointer import GIN4Ordering
import os

all_batch_size = {
    "nips": 128,
    "aan": 128,
    "nsf": 128,
    "sind": 128, 
    "roc": 128,
}

all_lr = {
    "nips": 1e-4,
    "aan": 5e-4,
    "nsf": 6e-4,
    "sind": 4e-4,
    "roc": 3e-4,
}

all_epoch = {
    "nips": 5,
    "aan": 5,
    "nsf": 5,
    "sind": 5,
    "roc": 5,
}

# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--is_training",
                    default=True,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--learning_rate",
                    default=5e-4,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--task",
                    default="sind",
                    type=str,
                    help="Task")
parser.add_argument("--is_finetuning",
                    default=False,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--save_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
data_path = "./data/" + args.task + "/"
result_path = "./output/" + args.task + "/"
args.save_path += GIN4Ordering.__name__ + "." + args.task
args.score_file_path = result_path + GIN4Ordering.__name__ + "." + args.task + "." + args.score_file_path
args.log_path += GIN4Ordering.__name__ + "." + args.task + ".log"
args.batch_size = all_batch_size[args.task]
args.learning_rate = all_lr[args.task]
args.epochs = all_epoch[args.task]

logging.basicConfig(filename=args.log_path, level=logging.INFO)
logger = logging.getLogger(__name__)

print(args)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_model():
    X_train_input_ids, X_train_attention_mask, X_train_token_type, X_train_story_len = torch.load(data_path + "cached_train")
    X_train_adj1 = torch.load(data_path + "train.adj1")
    X_train_adj2 = torch.load(data_path + "train.adj2")
    X_train_adj3 = torch.load(data_path + "train.adj3")
    y_train = torch.load(data_path + "output_label_train")
    X_train_input_ids, X_train_attention_mask, X_train_token_type, X_train_adj1, X_train_adj2, X_train_adj3, X_train_story_len, y_train = np.array(X_train_input_ids), np.array(X_train_attention_mask), np.array(X_train_token_type), np.array(X_train_adj1), np.array(X_train_adj2), np.array(X_train_adj3), np.array(X_train_story_len), np.array(y_train)

    X_val_input_ids, X_val_attention_mask, X_val_token_type, X_val_story_len = torch.load(data_path + "cached_test")
    X_val_adj1 = torch.load(data_path + "test.adj1")
    X_val_adj2 = torch.load(data_path + "test.adj2")
    X_val_adj3 = torch.load(data_path + "test.adj3")
    y_val = torch.load(data_path + "output_label_test")
    X_val_input_ids, X_val_attention_mask, X_val_token_type, X_val_adj1, X_val_adj2, X_val_adj3, X_val_story_len, y_val = np.array(X_val_input_ids), np.array(X_val_attention_mask), np.array(X_val_token_type), np.array(X_val_adj1), np.array(X_val_adj2), np.array(X_val_adj3), np.array(X_val_story_len), np.array(y_val)
    
    model = GIN4Ordering(args=args, logger=logger)
    if args.is_finetuning:
        for param in model.bert_encoder.parameters():
            param.requires_grad = True
    else:
        for param in model.bert_encoder.parameters():
            param.requires_grad = False
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)
    model.fit(
        X_train_input_ids, X_train_attention_mask, X_train_token_type, X_train_adj1, X_train_adj2, X_train_adj3, X_train_story_len, y_train, 
        X_val_input_ids, X_val_attention_mask, X_val_token_type, X_val_adj1, X_val_adj2, X_val_adj3, X_val_story_len, y_val
    )

def test_model():
    X_val_input_ids, X_val_attention_mask, X_val_token_type, X_val_story_len = torch.load(data_path + "cached_test")
    X_val_adj1 = torch.load(data_path + "test.adj1")
    X_val_adj2 = torch.load(data_path + "test.adj2")
    X_val_adj3 = torch.load(data_path + "test.adj3")
    y_val = torch.load(data_path + "output_label_test")
    X_val_input_ids, X_val_attention_mask, X_val_token_type, X_val_adj1, X_val_adj2, X_val_adj3, X_val_story_len, y_val = np.array(X_val_input_ids), np.array(X_val_attention_mask), np.array(X_val_token_type), np.array(X_val_adj1), np.array(X_val_adj2), np.array(X_val_adj3), np.array(X_val_story_len), np.array(y_val)
    model = GIN4Ordering(args=args, logger=logger)
    model.load_model(args.save_path + ".pt")
    model.evaluate(X_val_input_ids, X_val_attention_mask, X_val_token_type, X_val_adj1, X_val_adj2, X_val_adj3, X_val_story_len, y_val, is_test=True)

if __name__ == '__main__':
    start = time.time()
    set_seed()
    if args.is_training:
        train_model()
    else:
        print("test")
        test_model()
    end = time.time()
    print("use time: ", (end - start) / 60, " min")
