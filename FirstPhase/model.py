from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import csv
import codecs
import functools
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer)
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            save_steps = len(epoch_iterator) // 5
            if global_step % save_steps == 0:
                if args.evaluate_during_training:
                    results = evaluate(args, model, tokenizer)
                print('loss: ' + str((tr_loss - logging_loss)/save_steps) + ' step: ' + str(global_step))
                logging_loss = tr_loss
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step

class MyTestDataset(Dataset):
    def __init__(self, tensor_data, sents):
        assert len(tensor_data) == len(sents)
        self.tensor_data = tensor_data
        self.rows = sents
        
    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return (self.tensor_data[idx], self.rows[idx])

def evaluate_test(args, model, tokenizer, file_type, prefix=""):
    processor = PairProcessor(args.window_size)
    output_mode = "classification"
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(file_type, 'bert', str(args.max_seq_length), 'pair_order'))
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()

    examples, lines = processor.get_test_examples(args.data_dir, file_type)
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )
    logger.info("Saving features into cached file %s", cached_features_file)
    torch.save(features, cached_features_file)
    torch.save(lines, cached_features_file + '_lines')

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    eval_outputs_dirs = (args.output_dir,)
    file_h = codecs.open(args.data_dir + file_type + "_results." + str(args.window_size) + ".tsv", "w", "utf-8")
    outF = csv.writer(file_h, delimiter='\t')

    results = {}
    for eval_output_dir in eval_outputs_dirs:
        eval_dataset = MyTestDataset(dataset, lines)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            row = batch[1]
            rows = {
                'guid': row[0],
                'text_a': row[1],
                'text_b': row[2],
                'labels': row[3],
                'pos_a': row[4],
                'pos_b':row[5]
            }
            del row
            batch = tuple(t.to(args.device) for t in batch[0])
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            logits = logits.detach().cpu().numpy()
            tmp_pred = np.argmax(logits, axis=1)
            for widx in range(logits.shape[0]):
                outF.writerow([rows['guid'][widx], rows['text_a'][widx], rows['text_b'][widx], rows['labels'][widx], rows['pos_a'][widx], rows['pos_b'][widx], logits[widx][0], logits[widx][1], tmp_pred[widx]])
            if preds is None:
                preds = logits
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits, axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = compute_metrics("mnli", preds, out_label_ids)
        results.update(result)

        file_h.close()
        output_eval_file = os.path.join(eval_output_dir, file_type + "_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results

def evaluate(args, model, tokenizer, prefix=""):
    eval_outputs_dirs = (args.output_dir,)
    results = {}
    for eval_output_dir in eval_outputs_dirs:
        eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size)
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = compute_metrics("mnli", preds, out_label_ids)
        results.update(result)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results

def load_and_cache_examples(args, tokenizer, evaluate=False):
    processor = PairProcessor(args.window_size)
    output_mode = 'classification'
    cached_features_file = os.path.join(
        args.data_dir, 'cached_{}_{}_{}_{}'.format('dev' if evaluate else 'train',
        'bert', str(args.max_seq_length), 'pair_order'))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info(
            "Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                    tokenizer,
                                    label_list=label_list,
                                    max_length=args.max_seq_length,
                                    output_mode=output_mode,
                                    pad_on_left=False,
                                    pad_token=tokenizer.convert_tokens_to_ids(
                                        [tokenizer.pad_token])[0],
                                    pad_token_segment_id=0,
                                    )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

class PairProcessor(DataProcessor):
    def __init__(self, window_size):
        super(PairProcessor, self).__init__()
        self.window_size = window_size

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train." + str(self.window_size) + ".tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "val." + str(self.window_size) + ".tsv")), "dev")

    def get_test_examples(self, data_dir, file_type):
        return self._create_test_examples(self._read_tsv(os.path.join(data_dir, file_type + ".4predict." + str(self.window_size) + ".tsv")), file_type)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (_, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[1].lower()
                text_b = line[2].lower()
                label = line[3]
            except IndexError:
                print('cannot read the line: ' + line)
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,label=label))
        return examples

    def _create_test_examples(self, lines, set_type):
        examples, rows = [], []
        for (_, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[1].lower()
                text_b = line[2].lower()
                label = line[3]
            except IndexError:
                print('cannot read the line: ' + line)
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            rows.append(line)
        return examples, rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, 
                        type=str, required=True,
                        help="The input data dir. Should contain the .tsv "
                        "files (or other data files) for the task.")
    parser.add_argument("--output_dir", default='run_glue_test', 
                        type=str, required=True,
                        help="The output directory where the model "
                        "predictions and checkpoints will be written.")
    parser.add_argument("--window_size", type=int, required=True,
                        help="The window size.")                    
    parser.add_argument("--max_seq_length", default=105, type=int,
                        help="The maximum total input sequence length "
                        "after tokenization. Sequences longer than this "
                        "will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

    # Set seed
    set_seed(args)
    args.output_mode = "classification"
    args.model_type = 'bert'
    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
    model = model_class.from_pretrained('bert-base-uncased')
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    results = {}
    if args.do_eval or args.do_test:
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            if args.do_test:
                result = evaluate_test(args, model, tokenizer, "train", prefix=global_step)
                result = evaluate_test(args, model, tokenizer, "val", prefix=global_step)
                result = evaluate_test(args, model, tokenizer, "test", prefix=global_step)
            elif args.do_eval:
                result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    return results

if __name__ == "__main__":
    main()
