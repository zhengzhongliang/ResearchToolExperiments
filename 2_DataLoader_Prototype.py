from random import sample
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader


data_folder_path = "data"

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

    class OpenbookDataset(Dataset):
        """Face Landmarks dataset."""

        def __init__(self, train_list, dev_list, test_list):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """
            self.all_instances = train_list+dev_list+test_list

        def __len__(self):
            return len(self.all_instances)

        def __getitem__(self, idx):

            return self.all_instances[idx]

    def print_batch(sample_batched):
        print("="*20)
        print("id", sample_batched["id"])
        print("facts", sample_batched["facts"])
        print("text", sample_batched["text"])
        print("label", sample_batched["label"])
        print("documents", sample_batched["documents"])
        print("query", sample_batched["query"])
        print("facts", sample_batched["facts"])

        input("\n")

        return 0

    openbook_dataset = OpenbookDataset( train_list, dev_list, test_list)
    print("dataset built!")
    openbook_dataloader = DataLoader(openbook_dataset, batch_size=4,
                        shuffle=True, num_workers=1)
    print("dataloader built!")

    # question: what does this num_workers mean
    for i_batch, sample_batched in enumerate(openbook_dataloader):
        print("enter each batch")
        print_batch(sample_batched)

    return 0


def main():
    dataloader_test()

main()

# glove path: /Users/zhengzhongliang/NLP_Research/Glove_Embedding/glove.840B.300d.pickle


