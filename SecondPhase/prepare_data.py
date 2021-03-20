import json
import torch
import numpy as np
import random
from transformers import BertTokenizer

def load_json_file(split):
    data = json.load(open('../FirstPhase/data/sind/' + split + '.story-in-sequence.json','r'))
    return data

def get_story_text(data):
    story_sentences = {}
    annotations = data['annotations']
    for annotation in annotations:
        story_id = annotation[0]['story_id']
        story_sentences.setdefault(story_id, [])
        story_sentences[story_id].append(annotation[0]['original_text'])
    return story_sentences

def convert_to_features_all(directory, split, task, data_file, window_size, max_length=50, num_sampled_times=1):
    def get_filenames(split):
        with open(split, "r") as inp:
            filenames = inp.read()
        filenames = filenames.split('\n')[:-1]
        return filenames
    def get_roc_text(split):
        data = torch.load("../FirstPhase/data/roc/" + split + ".text")
        story_sentences = {}
        for d in data:
            story_sentences[d[0]] = d[1:]
        return story_sentences
    story_sentences = {}
    remove_story_id = {}
    story_one_sentence = []
    max_len = 0
    min_len = 100
    count = 0
    if task == "roc":
        story_sentences = get_roc_text(split)
        max_len = 5
        min_len = 5
    elif task == "sind":
        data = load_json_file(data_file)
        story_sentences = get_story_text(data)
        for story_id in story_sentences:
            if len(story_sentences[story_id]) > max_len:
                max_len = len(story_sentences[story_id])
            if len(story_sentences[story_id]) < min_len:
                min_len = len(story_sentences[story_id])
        story_one_sentence = []
    else:
        dpath = directory + 'split/' + split
        filenames = get_filenames(dpath)
        for afile in filenames:
            if task == 'nips':
                with open(directory + 'txt_tokenized/' + 'a' + afile + '.txt', 'r') as inp:
                    lines = inp.readlines()
            else:
                with open(directory + 'txt_tokenized/' + afile, 'r') as inp:
                    lines = inp.readlines()
            lines = [line.strip() for line in lines]
            if len(lines) > max_len and len(lines) <= 40:
                max_len = len(lines)
            if len(lines) < min_len:
                min_len = len(lines)
            if len(lines) > 40:
                remove_story_id[afile] = 1
            story_sentences[afile] = lines
            if len(lines) == 1:
                story_one_sentence.append(afile)
    print(max_len)
    print(min_len)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 0
    all_input_id, all_attention_mask, all_token_type, all_story_length = [], [], [], []
    labels = []
    for _ in range(num_sampled_times):
        story_id_dict = {}
        with open("../FirstPhase/data/" + task + "_data/" + data_file + ".4predict." + str(window_size) + ".tsv", "r") as fr:
            for line in fr:
                line = line.strip().split("\t")
                if 'sind' in task:
                    story_id = line[0].split("-")[0]
                elif task == "roc":
                    story_id = "-".join(line[0].split("-")[:-3])
                else:
                    story_id = line[0].split("~")[0]
                if story_id not in story_id_dict and story_id not in remove_story_id:
                    story_id_dict[story_id] = 1
                    whole_story = story_sentences[story_id]
                    story_length = len(whole_story)
                    a = list(range(story_length))
                    b = list(range(max_len))
                    random.shuffle(a)
                    tmp_input_id, tmp_attention_mask, tmp_token_type = [], [], []
                    for idx in a:
                        encoding = tokenizer.encode_plus(whole_story[idx].lower(), max_length=50, add_special_tokens=True)
                        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
                        attention_mask = [1] * len(input_ids)
                        padding_length = max_length - len(input_ids)
                        input_ids = input_ids + ([pad_token] * padding_length)
                        attention_mask = attention_mask + ([0] * padding_length)
                        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
                        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
                        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
                        tmp_input_id.append(input_ids)
                        tmp_attention_mask.append(attention_mask)
                        tmp_token_type.append(token_type_ids)
                    for _ in range(max_len - story_length):
                        encoding = tokenizer.encode_plus("", max_length=50, add_special_tokens=True)
                        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
                        attention_mask = [1] * len(input_ids)
                        padding_length = max_length - len(input_ids)
                        input_ids = input_ids + ([pad_token] * padding_length)
                        attention_mask = attention_mask + ([0] * padding_length)
                        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
                        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
                        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
                        tmp_input_id.append(input_ids)
                        tmp_attention_mask.append(attention_mask)
                        tmp_token_type.append(token_type_ids)
                    all_input_id.append(tmp_input_id)
                    all_attention_mask.append(tmp_attention_mask)
                    all_token_type.append(tmp_token_type)
                    all_story_length.append(story_length)
                    b[:story_length] = a
                    labels.append(b)
        for story_id in story_one_sentence:
            story_id_dict[story_id] = 1
            whole_story = story_sentences[story_id]
            story_length = len(whole_story)
            a = list(range(story_length))
            b = list(range(max_len))
            random.shuffle(a)
            tmp_input_id, tmp_attention_mask, tmp_token_type = [], [], []
            for idx in a:
                encoding = tokenizer.encode_plus(whole_story[idx].lower(), max_length=50, add_special_tokens=True)
                input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
                attention_mask = [1] * len(input_ids)
                padding_length = max_length - len(input_ids)
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
                assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
                assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
                tmp_input_id.append(input_ids)
                tmp_attention_mask.append(attention_mask)
                tmp_token_type.append(token_type_ids)
            for _ in range(max_len - story_length):
                encoding = tokenizer.encode_plus("", max_length=50, add_special_tokens=True)
                input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
                attention_mask = [1] * len(input_ids)
                padding_length = max_length - len(input_ids)
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
                assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
                assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
                tmp_input_id.append(input_ids)
                tmp_attention_mask.append(attention_mask)
                tmp_token_type.append(token_type_ids)
            all_input_id.append(tmp_input_id)
            all_attention_mask.append(tmp_attention_mask)
            all_token_type.append(tmp_token_type)
            all_story_length.append(story_length)
            b[:story_length] = a
            labels.append(b)
    torch.save((all_input_id, all_attention_mask, all_token_type, all_story_length), "./data/" + task + "/cached_" + data_file)
    torch.save(labels, "./data/" + task + "/label_" + data_file)
    print(len(labels))

def get_all_adj_matrix(label_file, task, split, window_sizes=[], num_sampled_times=1, directory=None):
    def get_remove_one_sent_story_id(split):
        def get_filenames(split):
            with open(split, "r") as inp:
                filenames = inp.read()
            filenames = filenames.split('\n')[:-1]
            return filenames
        dpath = directory + 'split/' + split
        filenames = get_filenames(dpath)
        remove_story_id = {}
        one_sent_story = []
        max_len = 0
        for afile in filenames:
            if task == 'nips':
                with open(directory + 'txt_tokenized/' + 'a' + afile + '.txt', 'r') as inp:
                    lines = inp.readlines()
            else:
                with open(directory + 'txt_tokenized/' + afile, 'r') as inp:
                    lines = inp.readlines()
            lines = [line.strip() for line in lines]
            if len(lines) > 40:
                remove_story_id[afile] = 1
            if len(lines) == 1:
                one_sent_story.append(afile)
        return remove_story_id, one_sent_story

    max_story_len = {"sind": {"train": 5, "test": 5, "val": 5}, "nips": {"train": 15, "test": 14}, "aan": {"train": 19, "test": 18}, "roc": {"train": 5, "test": 5}, "nsf": {"train": 40, "test": 40}}
    
    if task == "nips" or task == "sind" or task == "roc":
        remove_story_id, one_sent_story = {}, {}
    else:
        remove_story_id, one_sent_story = get_remove_one_sent_story_id(split)
    print("remove ids: ", len(remove_story_id))
    print("one sentences: ", len(one_sent_story))

    labels = torch.load(label_file)
    story_len = max_story_len[task][split]
    label_dict = {}

    all_adjs = []
    lid = 0
    adj_dict = {}
    sid_array = []
    with open("../FirstPhase/data/" + task + "_data/" + split + "_results." + window_sizes[0] + ".tsv", "r") as fr:
        for idx, line in enumerate(fr):
            line = line.strip().split("\t")
            if 'sind' in task:
                sid = line[0].split("-")[0]
            elif task == "roc":
                sid = "-".join(line[0].split("-")[:-3])
            else:
                sid = line[0].split("~")[0]
            if sid in remove_story_id:
                continue
            if sid not in adj_dict:
                tmp_adj_matrix = np.zeros((story_len, story_len))
                label = labels[lid]
                label_dict[sid] = label
                adj_dict[sid] = tmp_adj_matrix
                sid_array.append(sid)
                lid += 1
            map_dict = {label[i]: i for i in range(len(label))}
            sent1_id = map_dict[int(line[4])]
            sent2_id = map_dict[int(line[5])]
            first_score = float(line[6])
            second_score = float(line[7])
            all_score = torch.tensor([first_score, second_score]).data.numpy().tolist()
            first_score, second_score = all_score[0], all_score[1]
            if first_score > second_score:
                adj_dict[sid][sent2_id][sent1_id] = first_score
            else:
                adj_dict[sid][sent1_id][sent2_id] = second_score
        for sid in one_sent_story:
            tmp_adj_matrix = np.zeros((story_len, story_len))
            label = labels[lid]
            label_dict[sid] = label
            adj_dict[sid] = tmp_adj_matrix
            sid_array.append(sid)
            lid += 1
        for sid in sid_array:
            for i in range(story_len):
                for j in range(i + 1, story_len):
                    if adj_dict[sid][i, j] > adj_dict[sid][j, i]:
                        adj_dict[sid][j, i] = 0.0
                    else:
                        adj_dict[sid][i, j] = 0.0
            all_adjs.append(adj_dict[sid])
    print(len(all_adjs))
    print(adj_dict[sid_array[0]])
    torch.save(all_adjs, "./data/" + task + "/" + split + ".adj1")

    all_adjs2 = []
    adj_dict2 = {}
    # sid_array2 = []
    lid = 0
    with open("../FirstPhase/data/" + task + "_data/" + split + "_results." + window_sizes[1] + ".tsv", "r") as fr:
        for idx, line in enumerate(fr):
            line = line.strip().split("\t")
            if task == 'sind':
                sid = line[0].split("-")[0]
            elif task == "roc":
                sid = "-".join(line[0].split("-")[:-3])
            else:
                sid = line[0].split("~")[0]
            if sid not in adj_dict2:
                tmp_adj_matrix = np.zeros((story_len, story_len))
                adj_dict2[sid] = tmp_adj_matrix
                label = label_dict[sid]
            map_dict = {label[i]: i for i in range(len(label))}
            sent1_id = map_dict[int(line[4])]
            sent2_id = map_dict[int(line[5])]
            first_score = float(line[6])
            second_score = float(line[7])
            all_score = torch.tensor([first_score, second_score]).data.numpy().tolist()
            first_score, second_score = all_score[0], all_score[1]
            if first_score < second_score:
                adj_dict2[sid][sent1_id][sent2_id] = second_score
        for sid in one_sent_story:
            tmp_adj_matrix = np.zeros((story_len, story_len))
            adj_dict2[sid] = tmp_adj_matrix
        for idx, sid in enumerate(sid_array):
            for i in range(story_len):
                for j in range(i + 1, story_len):
                    if adj_dict2[sid][i, j] > adj_dict2[sid][j, i]:
                        adj_dict2[sid][j, i] = 0.0
                    else:
                        adj_dict2[sid][i, j] = 0.0
            all_adjs2.append(adj_dict2[sid])

    print(len(all_adjs2))
    print(adj_dict2[sid_array[0]])
    torch.save(all_adjs2, "./data/" + task + "/" + split + ".adj2")

    all_adjs3 = []
    adj_dict3 = {}
    lid = 0
    with open("../FirstPhase/data/" + task + "_data/" + split + "_results." + window_sizes[2] + ".tsv", "r") as fr:
        for idx, line in enumerate(fr):
            line = line.strip().split("\t")
            if task == 'sind':
                sid = line[0].split("-")[0]
            elif task == "roc":
                sid = "-".join(line[0].split("-")[:-3])
            else:
                sid = line[0].split("~")[0]
            if sid not in adj_dict3:
                tmp_adj_matrix = np.zeros((story_len, story_len))
                adj_dict3[sid] = tmp_adj_matrix
                label = label_dict[sid]
                lid += 1
            map_dict = {label[i]: i for i in range(len(label))}
            sent1_id = map_dict[int(line[4])]
            sent2_id = map_dict[int(line[5])]
            first_score = float(line[6])
            second_score = float(line[7])
            all_score = torch.tensor([first_score, second_score]).data.numpy().tolist()
            first_score, second_score = all_score[0], all_score[1]
            if first_score < second_score:
                adj_dict3[sid][sent1_id][sent2_id] = second_score
        for sid in one_sent_story:
            tmp_adj_matrix = np.zeros((story_len, story_len))
            adj_dict3[sid] = tmp_adj_matrix
            lid += 1
        for idx, sid in enumerate(sid_array):
            for i in range(story_len):
                for j in range(i + 1, story_len):
                    if adj_dict3[sid][i, j] > adj_dict3[sid][j, i]:
                        adj_dict3[sid][j, i] = 0.0
                    else:
                        adj_dict3[sid][i, j] = 0.0
            all_adjs3.append(adj_dict3[sid])

    print(len(all_adjs3))
    print(adj_dict3[sid_array[0]])
    assert len(all_adjs) == len(all_adjs2) == len(all_adjs3)
    torch.save(all_adjs3, "./data/" + task + "/" + split + ".adj3")

def generate_labels(task, data_type):
    label = torch.load("./data/" + task + "/label_" + data_type)
    output_label = []
    for x in label:
        output_label.append(np.argsort(x))
    torch.save(output_label, "./data/" + task + "/output_label_" + data_type)

# NIPS
convert_to_features_all("../FirstPhase/data/nips/", split="2013le_papers", task="nips", data_file="train", window_size=5)
convert_to_features_all("../FirstPhase/data/nips/", split="2015_papers", task="nips", data_file="test", window_size=5)
get_all_adj_matrix("./data/nips/label_train", "nips", "train", num_sampled_times=1, window_sizes=[5, 8, 15])
get_all_adj_matrix("./data/nips/label_test", "nips", "test", num_sampled_times=1, window_sizes=[5, 8, 15])
generate_labels("nips", "train")
generate_labels("nips", "test")

# AAN
convert_to_features_all("../FirstPhase/data/aan/", split="train", task="aan", data_file="train", num_sampled_times=1, window_size=6)
convert_to_features_all("../FirstPhase/data/aan/", split="test", task="aan", data_file="test", num_sampled_times=1, window_size=6)
get_all_adj_matrix("./data/aan/label_train", "aan", "train", directory="../FirstPhase/data/aan/", num_sampled_times=1, window_sizes=[6, 11, 20])
get_all_adj_matrix("./data/aan/label_test", "aan", "test", directory="../FirstPhase/data/aan/",num_sampled_times=1, window_sizes=[6, 11, 20])
generate_labels("aan", "train")
generate_labels("aan", "test")

# NSF
convert_to_features_all("../FirstPhase/data/nsf/", split="train", task="nsf", data_file="train", num_sampled_times=1, window_size=11)
convert_to_features_all("../FirstPhase/data/nsf/", split="test", task="nsf", data_file="test", num_sampled_times=1, window_size=11)
get_all_adj_matrix("./data/nsf/label_train", "nsf", "train", directory="../FirstPhase/data/nsf/", num_sampled_times=1, window_sizes=[11, 21, 40])
get_all_adj_matrix("./data/nsf/label_test", "nsf", "test", directory="../FirstPhase/data/nsf/",num_sampled_times=1, [11, 21, 40])
generate_labels("nsf", "train")
generate_labels("nsf", "test")

# SIND
convert_to_features_all("../FirstPhase/data/sind/", split="train", task="sind", data_file="train", window_size=2)
convert_to_features_all("../FirstPhase/data/sind/", split="test", task="sind", data_file="test", window_size=2)
get_all_adj_matrix("./data/sind/label_train", "sind", "train", num_sampled_times=1, window_sizes=[2, 3, 5])
get_all_adj_matrix("./data/sind/label_test", "sind", "test", num_sampled_times=1, window_sizes=[2, 3, 5])
generate_labels("sind", "train")
generate_labels("sind", "test")

# ROC
convert_to_features_all("../FirstPhase/data/sind/roc/", split="train", task="roc", data_file="train", window_size=2)
convert_to_features_all("../FirstPhase/data/sind/roc/", split="test", task="roc", data_file="test", window_size=2)
get_all_adj_matrix("./data/roc/label_train", "roc", "train", num_sampled_times=1, window_sizes=[2, 3, 5])
get_all_adj_matrix("./data/roc/label_test", "roc", "test", num_sampled_times=1, window_sizes=[2, 3, 5])
generate_labels("roc", "train")
generate_labels("roc", "test")