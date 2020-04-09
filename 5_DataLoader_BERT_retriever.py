from random import sample
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random
import os
import torch.optim as optim
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
            negative_indices = random_negative_from_kb([target_fact_num], kb_as_list, 4)
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


class BertRetriever(torch.nn.Module):
    def __init__(self):
        super(BertRetriever, self).__init__()
        self.bert_q = BertModel.from_pretrained('bert-base-uncased')
        self.bert_d = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, query_token_ids, query_seg_ids, fact_token_ids, fact_seg_ids):
        query_output_tensor_, _ = self.bert_q(query_token_ids, query_seg_ids)
        fact_output_tensor_, _ = self.bert_d(fact_token_ids, fact_seg_ids)

        batch_size = query_token_ids.size()[0]

        query_output_tensor = query_output_tensor_[-1][:, 0].view(batch_size, 768, 1)
        fact_output_tensor = fact_output_tensor_[-1][:, 0].view(batch_size, 5, 768)

        return query_output_tensor, fact_output_tensor


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
        max_len_query = max([len(sample["query_token_ids"]) for sample in batch]) # this should be equivalent to "for x in batch"
        # pad according to max_len
        for sample in batch:
            sample["query_token_ids"]  = pad_tensor(sample["query_token_ids"], pad=max_len_query)
        # stack all

        # the output of this function needs to be a already batched function.
        batch_returned = {}
        batch_returned["query_token_ids"] = torch.tensor([[101]+sample["query_token_ids"]+[102] for sample in batch])
        batch_returned["query_seg_ids"] = torch.tensor([[0]*(max_len_query+2) for sample in batch])

        all_facts_ids = []
        for sample in batch:
            all_facts_ids.extend(sample["fact_token_ids"])

        max_len_fact = max([len(fact_token_ids) for fact_token_ids in all_facts_ids])

        for i, fact_ids in enumerate(all_facts_ids):
            all_facts_ids[i]  = pad_tensor(fact_ids, pad=max_len_fact)
        # stack all

        # the output of this function needs to be a already batched function.
        batch_returned["fact_token_ids"] = torch.tensor([[101]+fact_ids+[102] for fact_ids in all_facts_ids])
        batch_returned["fact_seg_ids"] = torch.tensor([[0]*(max_len_fact+2) for fact_ids in all_facts_ids])

        batch_returned["label_in_distractor"] = torch.tensor([sample["label_in_distractor"] for sample in batch])

        return batch_returned

    def __call__(self, batch):
        return self.pad_collate(batch)

class OpenbookDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, instance_list, kb,  tokenizer):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.instance_list=  []
        for instance in instance_list:
            # cls_id = 101; sep_id = 102; pad_id = 0;
            query_tokens = tokenizer.tokenize(instance["text"])   # this is for strip quotes
            query_token_ids = tokenizer.convert_tokens_to_ids(query_tokens)   # this does not include pad, cls or sep

            fact_token_ids = []
            fact_seg_ids = []
            for fact_index in instance["documents"]:
                single_fact_tokens = tokenizer.tokenize(kb[fact_index][1:-1]) # this if for removing the quotes
                single_fact_token_ids = tokenizer.convert_tokens_to_ids(single_fact_tokens)
                fact_token_ids.append(single_fact_token_ids)
                fact_seg_ids.append([0]*len(single_fact_token_ids))

            instance["query_token_ids"] = query_token_ids
            instance["query_seg_ids"] = [0]*len(query_token_ids)
            instance["fact_token_ids"] = fact_token_ids
            instance["fact_seg_ids"] = fact_seg_ids

            instance["label_in_distractor"] = 0

        self.instance_list = instance_list

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):

        return self.instance_list[idx]

def forward_pass_epoch_naive(train_list, dev_list, test_list, kb, tokenizer, bert_model):
    print("=" * 20)
    print("\tuse naive loading")

    # train_data = OpenbookDataset(train_list, kb, tokenizer).instance_list
    # dev_data = OpenbookDataset(dev_list, kb, tokenizer).instance_list
    #
    # bert_model.eval()
    # save_array  = []
    #
    # start_time = time.time()
    # for instance in train_data:
    #     query_token_ids = torch.tensor([[101]+instance["query_token_ids"]+[102]]).to(device)
    #     query_seg_ids = torch.tensor([[0]+instance["query_seg_ids"]+[0]]).to(device)
    #     query_output_tensor, _ = bert_model(query_token_ids, query_seg_ids)
    #
    #     fact_outputs = []
    #     for i in range(len(instance["fact_token_ids"])):
    #         fact_token_ids = torch.tensor([[101] + instance["fact_token_ids"][i] + [102]]).to(device)
    #         fact_seg_ids = torch.tensor([[0] + instance["fact_seg_ids"][i] + [0]]).to(device)
    #         fact_output_tensor, _ = bert_model(fact_token_ids, fact_seg_ids)
    #
    #         fact_outputs.append(fact_output_tensor[-1][0,0])
    #
    #     fact_outputs = torch.stack(fact_outputs)
    #
    #     scores = torch.nn.functional.softmax()
    #
    #     final_tensor = output_tensor[-1][:, 0].detach().cpu().numpy()  # output[-1][:,0]: -1 means last layer, : means all batches, 0 means CLS embedding.
    #
    #     save_array.append(final_tensor)
    #
    # save_array = np.array(save_array)
    # print("\tarray size:", save_array.shape)
    # end_time = time.time()
    # np.save("test_arr", save_array)
    # return end_time-start_time

    return 0


def forward_pass_epoch_dataloader(train_list, dev_list, test_list, kb, tokenizer, bert_model, batch_size = 2):
    print("=" * 20)
    print("\tuse dataloader batch size ", batch_size)


    train_data = OpenbookDataset(train_list[:11], kb, tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                    shuffle=True, num_workers=3, collate_fn=PadCollate())

    dev_data = OpenbookDataset(dev_list[:11], kb, tokenizer)
    dev_dataloader = DataLoader(dev_data, batch_size=batch_size,
                                  shuffle=False, num_workers=3, collate_fn=PadCollate())

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(bert_model.parameters(), lr=0.00001)

    bert_model.train()

    start_time = time.time()

    total_loss = 0
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        query_output_tensor, fact_output_tensor = bert_model(batch["query_token_ids"].to(device), batch["query_seg_ids"].to(device), batch["fact_token_ids"].to(device), batch["fact_seg_ids"].to(device))

        scores = torch.matmul(fact_output_tensor, query_output_tensor).squeeze(dim=2)

        label = batch["label_in_distractor"].to(device)

        loss = criterion(scores, label)

        loss.backward()
        optimizer.step()

        total_loss+=loss.detach().cpu().numpy()

        if (i+1)%200==0:
            print("processing sample ", i, " average loss:", total_loss/i)

    end_time = time.time()

    print("total training time:", end_time-start_time)


    bert_model.eval()
    total_loss = 0
    correct_count = 0
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            query_output_tensor, fact_output_tensor = bert_model(batch["query_token_ids"].to(device),
                                                                 batch["query_seg_ids"].to(device),
                                                                 batch["fact_token_ids"].to(device),
                                                                 batch["fact_seg_ids"].to(device))


            scores = torch.matmul(fact_output_tensor, query_output_tensor).squeeze(dim=2)   # size of scores: [2,5]

            label =batch["label_in_distractor"].to(device)    # size of labels: [2]
            loss = criterion(scores, label)

            total_loss += loss.detach().cpu().numpy()

            _, pred_fact = torch.max(scores, dim=1)
            correct_count += torch.sum(pred_fact==label).detach().cpu().numpy()

        print("avg eval loss:", total_loss/len(dev_dataloader)/batch_size, " accuracy:", correct_count/len(dev_dataloader)/batch_size)

    return end_time - start_time


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertRetriever()
    bert_model.to(device)

    train_list, dev_list, test_list, kb = construct_retrieval_dataset_openbook()
    # time_naive = forward_pass_epoch_naive(kb, tokenizer, bert_model)
    # print("time naive:", time_naive)
    time_loader = forward_pass_epoch_dataloader(train_list, dev_list, test_list, kb, tokenizer, bert_model, batch_size=2)
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
