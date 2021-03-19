import torch
import csv
import json
import random
import argparse
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer, BertForSequenceClassification)


class DataHandler:
    def __init__(self, directory, task_name, window_size):
        for _, tokenizer_class, pretrained_weights in [(BertConfig, BertTokenizer, 'bert-base-uncased')]:
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.directory = directory
        self.task = task_name
        self.window_size = window_size
            
    def get_filenames(self, split):
        with open(split, "r") as inp:
            filenames = inp.read()
        filenames = filenames.split('\n')[:-1]
        return filenames

    def load_json_file(self, split):
        data = json.load(open(self.directory + split + '.story-in-sequence.json','r'))
        return data

    def get_story_text(self, data):
        story_sentences = {}
        annotations = data['annotations']
        for annotation in annotations:
            story_id = annotation[0]['story_id']
            story_sentences.setdefault(story_id, [])
            story_sentences[story_id].append(annotation[0]['original_text'])
        return story_sentences
    
    def get_roc_text(self, split):
        data = torch.load("../data/roc/" + split + ".text")
        story_sentences = {}
        for d in data:
            story_sentences[d[0]] = d[1:]
        return story_sentences
    
    def truncate_test(self, sent1, sent2):
        s1i = self.tokenizer.encode(sent1, add_special_tokens=False)
        s2i = self.tokenizer.encode(sent2, add_special_tokens=False)
        if len(s1i) < 50:
            sent2 = self.tokenizer.decode(s2i[:100-len(s1i)])
        elif len(s2i) < 50:
            sent1 = self.tokenizer.decode(s1i[:100-len(s2i)])
        else:
            sent1 = self.tokenizer.decode(s1i[:50])
            sent2 = self.tokenizer.decode(s2i[:50])
        inp = self.tokenizer.encode(sent1, sent2, add_special_tokens=True, max_length=104)
        assert len(inp) < 105
        return sent1, sent2
               
    def get_convert_write(self, split, filename, out_dir):
        dpath = self.directory + 'split/' + split
        filenames = self.get_filenames(dpath)
        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            for file in filenames:
                if self.task == 'nips':
                    with open(
                        self.directory + 'txt_tokenized/' + 'a' + file + '.txt', 
                        'r') as inp:
                        lines = inp.readlines()
                else:
                    with open(self.directory + 'txt_tokenized/' + file, 'r') as inp:
                        lines = inp.readlines()
                lines = [line.strip() for line in lines]
                if len(lines) > 40:
                    continue
                y += 1
                if y % 100 == 0:
                    print(y, x)
                pos = []
                neg = []
                for i in range(len(lines)):
                    for j in range(len(lines)):  
                        if i == j:
                            continue                
                        sent1 = lines[i]
                        sent2 = lines[j]
                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(sent1.lower(), sent2.lower(), add_special_tokens=True)
                        length = len(inputs)
                        if length > 100:
                            sent1, sent2 = self.truncate_test(sent1, sent2)
                        x += 1
                        if 0 < j - i < self.window_size:
                            pos.append([split+'-'+str(y)+'-'+str(x), sent1, sent2, 1])
                        else:
                            neg.append([split+'-'+str(y)+'-'+str(x), sent1, sent2, 0])
                neg = random.sample(neg, len(pos))
                for p, n in zip(pos, neg):
                    tsv_writer.writerow(p)
                    tsv_writer.writerow(n)

    def get_convert_write_sind(self, split, filename, out_dir):
        data = self.load_json_file(split)
        story_sentences = self.get_story_text(data)
        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            for story_id in story_sentences.keys():
                y += 1                
                if y % 100 == 0:
                    print(y, x) 
                story = story_sentences[story_id]
                pos, neg = [], []
                for i in range(len(story)):
                    for j in range(len(story)):
                        if j == i:
                            continue
                        sent1 = story[i]
                        sent2 = story[j]
                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(sent1.lower(), sent2.lower(), add_special_tokens=True)
                        length = len(inputs)
                        if length > 100:
                            sent1, sent2 = self.truncate_test(sent1, sent2)
                        x += 1
                        if 0 < j - i < self.window_size:
                            pos.append([split+'-'+str(y)+'-'+str(x), sent1, sent2, 1])
                        else:
                            neg.append([split+'-'+str(y)+'-'+str(x), sent1, sent2, 0])
                neg = random.sample(neg, len(pos))
                for p, n in zip(pos, neg):
                    tsv_writer.writerow(p)
                    tsv_writer.writerow(n)

    def get_convert_write_roc(self, split, filename, out_dir):
        story_sentences = self.get_roc_text(split)
        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            for story_id in story_sentences.keys():
                y += 1                
                if y % 100 == 0:
                    print(y, x) 
                story = story_sentences[story_id]
                pos, neg = [], []
                for i in range(len(story)):
                    for j in range(len(story)):
                        if j == i:
                            continue
                        sent1 = story[i]
                        sent2 = story[j]
                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(sent1.lower(), sent2.lower(), add_special_tokens=True)
                        length = len(inputs)
                        if length > 100:
                            sent1, sent2 = self.truncate_test(sent1, sent2)
                        x += 1
                        if 0 < j - i < self.window_size:
                            pos.append([split+'-'+str(y)+'-'+str(x), sent1, sent2, 1])
                        else:
                            neg.append([split+'-'+str(y)+'-'+str(x), sent1, sent2, 0])
                neg = random.sample(neg, len(pos))
                for p, n in zip(pos, neg):
                    tsv_writer.writerow(p)
                    tsv_writer.writerow(n)

    def write_test(self, split, filename, out_dir):
        dpath = self.directory + 'split/' + split
        filenames = self.get_filenames(dpath)
        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            for file in filenames:
                if self.task == 'nips':
                    with open(self.directory + 'txt_tokenized/' + 'a' + file + '.txt', 'r') as inp:
                        lines = inp.readlines()
                else:
                    with open(self.directory + 'txt_tokenized/' + file, 'r') as inp:
                        lines = inp.readlines()
                lines = [line.strip() for line in lines]
                if len(lines) > 40:
                    continue
                y += 1
                if y % 100 == 0:
                    print(y, x)
                tmp = []
                for i in range(len(lines)):
                    for j in range(len(lines)):                  
                        if i == j:
                            continue
                        sent1 = lines[i].lower()
                        sent2 = lines[j].lower()
                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(sent1, sent2, add_special_tokens=True)
                        length = len(inputs)
                        if length > 100:
                            sent1, sent2 = self.truncate_test(sent1, sent2)
                        x += 1
                        r = random.random()
                        if 0 < j - i < self.window_size:
                            tmp.append([file + "~" + str(y)+'-'+str(len(lines)), sent1, sent2, 1, i, j])
                        else:
                            tmp.append([file + "~" + str(y)+'-'+str(len(lines)), sent1, sent2, 0, i, j])
                for row in tmp:
                    # adding no of pairs of sentences in the end
                    row[0] += '-' + str(len(tmp))
                    tsv_writer.writerow(row)

    def write_test_sind(self, split, filename, out_dir):
        data = self.load_json_file(split)
        story_sentences = self.get_story_text(data)
 
        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            for story_id in story_sentences.keys():
                y += 1
                if y % 100 == 0:
                    print(y, x) 
                story = story_sentences[story_id]
                tmp = []
                pos, neg = [], []
                for i in range(len(story)):
                    for j in range(len(story)):
                        if j == i:
                            continue
                        sent1 = story[i]
                        sent2 = story[j]
                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(sent1.lower(), sent2.lower(), add_special_tokens=True)
                        length = len(inputs)
                        if length > 100:
                            sent1, sent2 = self.truncate_test(sent1, sent2)
                        x += 1
                        if 0 < j - i < self.window_size:
                            tmp.append([str(story_id)+'-'+str(y)+'-'+str(len(story)), sent1, sent2, 1, i, j])
                        else:
                            tmp.append([str(story_id)+'-'+str(y)+'-'+str(len(story)), sent1, sent2, 0, i, j])
                for row in tmp:
                    # adding no of pairs of sentences in the end
                    row[0] += '-' + str(len(tmp))
                    tsv_writer.writerow(row)
    
    def write_test_roc(self, split, filename, out_dir):
        story_sentences = self.get_roc_text(split)
        x, y = 0, 0
        filename = out_dir + filename
        with open(filename, "w") as out:
            tsv_writer = csv.writer(out, delimiter='\t')
            for story_id in story_sentences.keys():
                y += 1                
                if y % 100 == 0:
                    print(y, x) 
                story = story_sentences[story_id]
                tmp = []
                for i in range(len(story)):
                    for j in range(len(story)):
                        if j == i:
                            continue
                        sent1 = story[i]
                        sent2 = story[j]
                        #check if tokenized input is greater than 100
                        inputs = self.tokenizer.encode(sent1.lower(), sent2.lower(), add_special_tokens=True)
                        length = len(inputs)
                        if length > 100:
                            sent1, sent2 = self.truncate_test(sent1, sent2)
                        x += 1
                        if 0 < j - i < self.window_size:
                            tmp.append([str(story_id)+'-'+str(y)+'-'+str(len(story)), sent1, sent2, 1, i, j])
                        else:
                            tmp.append([str(story_id)+'-'+str(y)+'-'+str(len(story)), sent1, sent2, 0, i, j])
                for row in tmp:
                    #adding no of pairs of sentences in the end
                    row[0] += '-' + str(len(tmp))
                    tsv_writer.writerow(row)

        
def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str,
                         required=True, help="The input data dir.")
    parser.add_argument("--out_dir", default='', type=str,
                         help="The dir to save the output files.")
    parser.add_argument("--task_name", default='', type=str, required=True,
                         help="Task name can be nips | nsf | aan | sind")
    args = parser.parse_args()

    handler = DataHandler(args.data_dir, args.task_name, window_size=3)
    if args.task_name == 'sind':
        for window_size in [2, 3, 5]:
            handler.get_convert_write_sind('train', 'train.' + str(window_size) + '.tsv', args.out_dir)
            handler.get_convert_write_sind('val', 'val.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test_sind('test', 'test.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test_sind('train', 'train.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test_sind('val', 'val.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test_sind('test', 'test.4predict.' + str(window_size) + '.tsv', args.out_dir)
    elif args.task_name == "roc":
        for window_size in [2, 3, 5]:
            handler = DataHandler(args.data_dir, args.task_name, window_size=window_size)
            handler.get_convert_write_roc('train', 'train.' + str(window_size) + '.tsv', args.out_dir)
            handler.get_convert_write_roc('dev', 'val.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test_roc('test', 'test.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test_roc('train', 'train.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test_roc('dev', 'val.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test_roc('test', 'test.4predict.' + str(window_size) + '.tsv', args.out_dir)
    elif args.task_name == "nips":
        for window_size in [5, 8, 15]:
            handler.get_convert_write('2013le_papers', 'train.' + str(window_size) + '.tsv', args.out_dir)
            handler.get_convert_write('2014_papers', 'val.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('2015_papers', 'test.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('2013le_papers', 'train.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('2014_papers', 'val.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('2015_papers', 'test.4predict.' + str(window_size) + '.tsv', args.out_dir)
    elif args.task_name == "aan":
        for window_size in [6, 11, 20]:
            handler.get_convert_write('train', 'train.' + str(window_size) + '.tsv', args.out_dir)
            handler.get_convert_write('valid', 'val.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('test', 'test.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('train', 'train.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('valid', 'val.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('test', 'test.4predict.' + str(window_size) + '.tsv', args.out_dir)
    else:
        for window_size in [11, 21, 40]:
            handler.get_convert_write('train', 'train.' + str(window_size) + '.tsv', args.out_dir)
            handler.get_convert_write('valid', 'val.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('test', 'test.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('train', 'train.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('valid', 'val.4predict.' + str(window_size) + '.tsv', args.out_dir)
            handler.write_test('test', 'test.4predict.' + str(window_size) + '.tsv', args.out_dir)
    
if __name__ == "__main__":
    main()
