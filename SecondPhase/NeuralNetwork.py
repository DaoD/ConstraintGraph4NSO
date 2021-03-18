import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import Dataset
from tqdm import tqdm
from Evaluate import evaluate_accuracy, evaluate_pmr, evaluate_lcs, evaluate_tau, evaluate_first_last_accuracy
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.patience = 0
        self.init_clip_max_norm = 1.0
        self.optimizer = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.best_result = 0.0

    def forward(self):
        raise NotImplementedError

    def listMLE_loss(self, y_pred, y_label, story_len):
        """Loss for listMLE

        Args:
            y_pred ([Tensor]): [batch, num]
            y_label ([Tensor]): [batch, num]
            story_len ([Tensor]): [batch]
        """
        prod = []
        sort_y_pred = y_pred.gather(1, y_label)
        for i in range(y_pred.size(1)):
            pred = sort_y_pred[:, i:]
            pred_softmax = F.softmax(pred, dim=1)
            prod.append(pred_softmax[:, 0])
        prod = torch.stack(prod, dim=1) # [batch, num]
        story_len = story_len.unsqueeze(1)
        mask = torch.arange(prod.size(1)).cuda().unsqueeze(0).expand(*y_pred.size()) >= story_len
        float_mask = mask.float()
        prod[mask] = float_mask[mask]
        all_prod = 1.0
        for i in range(prod.size(1)):
            all_prod *= prod[:, i]
        all_prod = (all_prod + 1e-8).log().mean()
        return -all_prod

    def train_step(self, data):
        with torch.no_grad():
            batch_i, batch_a, batch_t, batch_adj1, batch_adj2, batch_adj3, batch_l, batch_y = (item.cuda(device=self.device) for item in data)
        # output: [batch, num_sentences]
        output, _ = self.forward(batch_i, batch_a, batch_t, batch_adj1, batch_adj2, batch_adj3, batch_l, is_train=True)
        loss = self.listMLE_loss(output, batch_y, batch_l)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.zero_grad()
        return loss, batch_y.size(0)

    def fit(self, X_train_input_ids, X_train_attention_mask, X_train_token_type_ids, X_train_adj1, X_train_adj2, X_train_adj3, X_train_story_len, y_train, X_dev_input_ids, X_dev_attention_mask, X_dev_token_type_ids, X_dev_adj1, X_dev_adj2, X_dev_adj3, X_dev_story_len, y_dev):
        if torch.cuda.is_available():
            self.cuda()
        dataset = Dataset(X_train_input_ids, X_train_attention_mask, X_train_token_type_ids, X_train_adj1, X_train_adj2, X_train_adj3, X_train_story_len, y_train)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        self.optimizer = AdamW(self.parameters(), eps=1e-8, lr=self.args.learning_rate)
        t_total = int(len(X_train_input_ids) * self.args.epochs // self.args.batch_size)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(t_total * 0.2), num_training_steps=t_total)
        one_epoch_step = len(y_train) // self.args.batch_size

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            avg_loss = 0
            all_len = len(dataloader)
            self.train()
            with tqdm(total=len(y_train), ncols=120) as pbar:
                for i, data in enumerate(dataloader):
                    loss, batch_size = self.train_step(data)
                    for param_group in self.optimizer.param_groups:
                        self.args.learning_rate = param_group['lr']
                    pbar.set_postfix(lr=self.args.learning_rate, loss=loss.item())

                    if i > 0 and i % (one_epoch_step // 5) == 0:
                        self.evaluate(X_dev_input_ids, X_dev_attention_mask, X_dev_token_type_ids, X_dev_adj1, X_dev_adj2, X_dev_adj3, X_dev_story_len, y_dev)
                        self.train()

                    if self.init_clip_max_norm is not None:
                        utils.clip_grad_norm_(self.parameters(), self.init_clip_max_norm)
                    pbar.update(batch_size)
                    avg_loss += loss.item()
            cnt = len(y_train) // self.args.batch_size + 1
            tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
            self.evaluate(X_dev_input_ids, X_dev_attention_mask, X_dev_token_type_ids, X_dev_adj1, X_dev_adj2, X_dev_adj3, X_dev_story_len, y_dev)

    def adjust_learning_rate(self, decay_rate=0.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']

    def evaluate(self, X_dev_input_ids, X_dev_attention_mask, X_dev_token_type_ids, X_dev_adj1, X_dev_adj2, X_dev_adj3, X_dev_story_len, y_dev, is_test=False):
        y_pred = self.predict(X_dev_input_ids, X_dev_attention_mask, X_dev_token_type_ids, X_dev_adj1, X_dev_adj2, X_dev_adj3, X_dev_story_len, y_dev)
        accuracy = evaluate_accuracy(y_pred, y_dev, X_dev_story_len)
        pmr, pmr_list = evaluate_pmr(y_pred, y_dev, X_dev_story_len, return_list=True)
        lcs = evaluate_lcs(y_pred, y_dev, X_dev_story_len)
        tau, tau_list = evaluate_tau(y_pred, y_dev, X_dev_story_len, return_list=True)
        first, last = evaluate_first_last_accuracy(y_pred, y_dev, X_dev_story_len)
        if not is_test and pmr + tau > self.best_result:
            self.best_result = pmr + tau
            tqdm.write("Best Result: PMR: %.4f, Acc %.4f, tau: %.4f, LCS: %.4f, first: %.4f, last:%.4f" % (pmr, accuracy, tau, lcs, first, last))
            self.logger.info("Best Result: PMR: %.4f, Acc %.4f, tau: %.4f, LCS: %.4f, first: %.4f, last:%.4f" % (pmr, accuracy, tau, lcs, first, last))
            self.patience = 0
            torch.save(self.state_dict(), self.args.save_path + ".pt")
            with open(self.args.score_file_path, 'w') as output:
                for tau, pmr in zip(tau_list, pmr_list):
                    output.write(str(tau) + '\t' + str(pmr) + '\n')
        else:
            self.patience += 1

        if is_test:
            tqdm.write("Best Result: PMR: %.4f, Acc %.4f, tau: %.4f, LCS: %.4f, first: %.4f, last:%.4f" % (pmr, accuracy, tau, lcs, first, last))

    def predict(self, X_dev_input_ids, X_dev_attention_mask, X_dev_token_type_ids, X_dev_adj1, X_dev_adj2, X_dev_adj3, X_dev_story_len, y_dev):
        self.eval()
        y_pred = []
        dataset = Dataset(X_dev_input_ids, X_dev_attention_mask, X_dev_token_type_ids, X_dev_adj1, X_dev_adj2, X_dev_adj3, X_dev_story_len)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size)
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                batch_i, batch_a, batch_t, batch_adj1, batch_adj2, batch_adj3, batch_l = (item.cuda() for item in data)
                _, y_pred_pointer = self.forward(batch_i, batch_a, batch_t, batch_adj1, batch_adj2, batch_adj3, batch_l, is_train=False)
                y_pred.append(y_pred_pointer.data.cpu().numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        return y_pred

    def load_model(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available():
            self.cuda()
