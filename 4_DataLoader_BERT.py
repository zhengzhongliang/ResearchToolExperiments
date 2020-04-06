from random import sample
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel


data_folder_path = "data"

# Here is a way to pad BERT: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# But it seems to be a very primitive way to pad it.

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# if os.path.exists("/Users/zhengzhongliang/NLP_Research/Glove_Embedding/glove.840B.300d.pickle"):
#     with open("/Users/zhengzhongliang/NLP_Research/Glove_Embedding/glove.840B.300d.pickle", "rb") as handle:
#         glove_dict = pickle.load(handle)
# else:
#     with open("/home/zhengzhongliang/CLU_Projects/glove.840B.300d.pickle", "rb") as handle:
#         glove_dict = pickle.load(handle)
#
#

def random_negative_from_kb(target_fact_num_list, kb_as_list, num_of_negative_facts):
    candidate_indexes = list(range(len(kb_as_list)))
    candidate_indexes_new = [x for x in candidate_indexes if x not in target_fact_num_list]
    selected_indexes = sample(candidate_indexes_new,num_of_negative_facts)

    return selected_indexes


def get_knowledge_base(kb_path: str):
    kb_data = list([])
    with open(kb_path, 'r') as the_file:
        kb_data = [line.strip() for line in the_file.readlines()]

    return kb_data


# Load questions as list of json files
def load_questions_json(question_path: str):
    questions_list = list([])
    with open(question_path, 'r', encoding='utf-8') as dataset:
        for i, line in enumerate(dataset):
            item = json.loads(line.strip())
            questions_list.append(item)

    return questions_list

def construct_dataset(train_path: str, dev_path: str, test_path: str, fact_path: str) -> (list, list, list):
    # This function is used to generate instances list for train, dev and test.
    def file_to_list(file_path: str, sci_facts: list) -> list:
        choice_to_id = {"A": 0, "B": 1, "C": 2, "D": 3}
        json_list = load_questions_json(file_path)

        instances_list = list([])
        for item in json_list:
            instance = {}
            instance["id"] = item["id"]
            for choice_id in range(4):
                if choice_id == choice_to_id[item['answerKey']]:
                    instance["text"] = item["question"]["stem"] + " " + item["question"]["choices"][choice_id]["text"]
                    gold_sci_fact = '\"' + item["fact1"] + '\"'
                    instance["label"] = sci_facts.index(gold_sci_fact)
            instances_list.append(instance)

        return instances_list

    sci_facts = get_knowledge_base(fact_path)

    train_list = file_to_list(train_path, sci_facts)
    dev_list = file_to_list(dev_path, sci_facts)
    test_list = file_to_list(test_path, sci_facts)

    return train_list, dev_list, test_list, sci_facts

def construct_retrieval_dataset_openbook():
    train_path = data_folder_path+"/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl"
    dev_path = data_folder_path+"/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl"
    test_path = data_folder_path+"/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl"
    fact_path = data_folder_path+"/OpenBookQA-V1-Sep2018/Data/Main/openbook.txt"

    # Build model:
    # Construct dataset
    train_raw, dev_raw, test_raw, sci_kb = construct_dataset(train_path, dev_path, test_path, fact_path)

    def add_distractor(instances_list, kb_as_list):
        instances_list_new = list([])
        for instance in instances_list:
            target_fact_num = instance["label"]
            negative_indices = random_negative_from_kb([target_fact_num], kb_as_list, 10)
            instance["documents"] = [target_fact_num]+negative_indices
            instance["query"] = [instance["text"]]
            instance["facts"] = [target_fact_num]
            instances_list_new.append(instance)

        return instances_list_new

    train_list = add_distractor(train_raw, sci_kb)
    dev_list = add_distractor(dev_raw, sci_kb)
    test_list = add_distractor(test_raw, sci_kb)

    print("openbook data constructed! train size:", len(train_list),"\tdev size:", len(dev_list),"\tkb size:", len(sci_kb))

    return train_list, dev_list, test_list, sci_kb

def pad_tensor(vec, pad):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """

    return vec + [0]*(pad-len(vec))


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """

        # The input here is actually a list of dictionary.
        # find longest sequence
        max_len = max([len(sample["token_ids"]) for sample in batch]) # this should be equivalent to "for x in batch"
        # pad according to max_len
        for sample in batch:
            sample["token_ids"]  = pad_tensor(sample["token_ids"], pad=max_len)
        # stack all

        # the output of this function needs to be a already batched function.
        batch_returned = {}
        batch_returned["token_ids"] = torch.tensor([[101]+sample["token_ids"]+[102] for sample in batch])
        batch_returned["seg_ids"] = torch.tensor([[0]*(max_len+2) for sample in batch])
        return batch_returned

    def __call__(self, batch):
        return self.pad_collate(batch)

class OpenbookDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, kb, tokenizer):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_instances = []
        for sci_fact in kb:
            # cls_id = 101; sep_id = 102; pad_id = 0;
            tokens = tokenizer.tokenize(sci_fact[1:-1])   # this is for strip quotes
            token_ids = tokenizer.convert_tokens_to_ids(tokens)   # this does not include pad, cls or sep
            self.all_instances.append({"token_ids":token_ids, "seg_ids":[0]*len(token_ids)})  # list of ids, list of ids.

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, idx):

        return self.all_instances[idx]

def get_list_dict_time():
    train_list, dev_list, test_list, kb = construct_retrieval_dataset_openbook()

    start_time = time.time()
    for i in range(5):
        for instance in train_list+test_list+dev_list:
            a= instance["id"]
            b= instance["text"]
            c= instance["label"]
            d= instance["documents"]
            e= instance["query"]
            f= instance["facts"]

    return (time.time()-start_time)/5

def dataloader_test():
    train_list, dev_list, test_list, kb = construct_retrieval_dataset_openbook()

    def print_batch(sample_batched):
        print("="*20)
        print("id", sample_batched["id"])
        print("tokens", sample_batched["tokens"])
        input("\n")

        return 0

    openbook_dataset = OpenbookDataset( train_list, dev_list, test_list)
    print("dataset built!")
    openbook_dataloader = DataLoader(openbook_dataset, batch_size=4,
                        shuffle=True, num_workers=1, collate_fn=PadCollate())
    print("dataloader built!")

    # question: what does this num_workers mean
    for i_batch, sample_batched in enumerate(openbook_dataloader):
        print("enter each batch")
        print("="*20)
        print(sample_batched["embds"].size())
        input("AAA")

    return 0

def forward_pass_epoch_naive(kb, tokenizer, bert_model):
    print("=" * 20)
    print("\tuse naive loading")

    openbook_dataset = OpenbookDataset(kb, tokenizer)
    all_instances = openbook_dataset.all_instances

    bert_model.eval()

    start_time = time.time()
    with torch.no_grad():
        for instance in all_instances:
            token_ids = torch.tensor([[101]+instance["token_ids"]+[102]]).to(device)
            seg_ids = torch.tensor([[0]+instance["seg_ids"]+[0]]).to(device)
            _ = bert_model(token_ids, seg_ids)

    end_time = time.time()
    return end_time-start_time


def forward_pass_epoch_dataloader(kb, tokenizer, bert_model, batch_size = 4):
    print("=" * 20)
    print("\tuse dataloader batch size ", batch_size)


    openbook_dataset = OpenbookDataset(kb, tokenizer)
    openbook_dataloader = DataLoader(openbook_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=3, collate_fn=PadCollate())

    bert_model.eval()

    start_time = time.time()
    with torch.no_grad():
        for batch in openbook_dataloader:

            _ = bert_model(batch["token_ids"].to(device), batch["seg_ids"].to(device))

    end_time = time.time()
    return end_time - start_time


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    train_list, dev_list, test_list, kb = construct_retrieval_dataset_openbook()
    time_naive = forward_pass_epoch_naive(kb, tokenizer, bert_model)
    print("time naive:", time_naive)
    time_loader = forward_pass_epoch_dataloader(kb, tokenizer, bert_model, batch_size=8)
    print("time loader:", time_loader)


main()

# glove path: /Users/zhengzhongliang/NLP_Research/Glove_Embedding/glove.840B.300d.pickle

# the original customized padding function:
'''
def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = map(lambda (x, y):
                    (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
'''
