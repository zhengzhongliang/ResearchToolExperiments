import re

def question_list_to_dict(question_list_list):
    print("\tProcessing raw dataset for later use ...")
    question_list_dict = list([])
    knowledge_base = list([])
    knowledge_base_dict = {}
    fact_id = 0

    fact_type = {}

    for question_num, question_list in enumerate(question_list_list):
        question_dict = {}
        question_dict['id'] = int(question_list[0][10:])
        question_dict['num'] = question_num
        question_stem_choice_raw = question_list[1].split('\t')
        question_dict['stem'] = question_stem_choice_raw[0][10:]
        question_dict['choices'] = [question_stem_choice_raw[i][5:] for i in range(1, 5)]
        question_dict['facts'] = list([])
        question_dict['answer'] = int(question_list[2].split(': ')[1])

        # capture the explanations of the questions
        explanations = question_list[4:]
        for explanation_raw in explanations:
            explanation = explanation_raw.split('(UID')[0]
            explanation_type = re.search(r'(?<=ROLE: )\w+', explanation_raw).group(0)

            if explanation not in knowledge_base_dict:
                knowledge_base_dict[explanation] = fact_id
                fact_type[fact_id] = explanation_type
                fact_id += 1


            #question_dict['facts'].append(knowledge_base_dict[explanation])

            if explanation_type=="CENTRAL" or explanation_type=="GROUNDING" or explanation_type=="ROLE":
                question_dict['facts'].append(knowledge_base_dict[explanation])
            # elif random.uniform(0, 1)>0.7:
            #     question_dict['facts'].append(knowledge_base_dict[explanation])

        if question_dict["answer"] != -1:
            question_list_dict.append(question_dict)

    # Generate the knowledge fact list from the dictionary:
    reversed_kb_dict = {v: k for k, v in knowledge_base_dict.items()}
    for i in reversed_kb_dict.keys():
        knowledge_base.append(reversed_kb_dict[i])

    return question_list_dict, knowledge_base, fact_type


data_path = "Data/explanations_plaintext.withmercury.txt"
with open(data_path, 'r') as raw_file:
    raw_text = raw_file.read()

raw_text = raw_text.split('\n\n\n')[:-1]  # this is to remove the \n character in the last

# convert each question to a list
question_list_list = [question.split('\n') for question in raw_text]


question_dict_list, kb, fact_type = question_list_to_dict(question_list_list)

chain_len_dict = {}
for question_dict in question_dict_list:
    if len(question_dict["facts"]) not in chain_len_dict:
        chain_len_dict[len(question_dict["facts"])]=1
    else:
        chain_len_dict[len(question_dict["facts"])] += 1

print("reasoning chain distribution:")
for i in sorted(chain_len_dict.keys()):
    print("len:",i, " num:", chain_len_dict[i])

for question_dict in question_dict_list:
    if len(question_dict["facts"])<=4:
        print("="*20)
        print(question_dict["stem"])
        for i in range(4):
            print("\t", question_dict["choices"][i])

        input("press enter to get answer:")
        print('answer:', question_dict["answer"])
        print('-'*20)
        print("supporting facts:")
        for index in question_dict["facts"]:
            print("\t", kb[index], " ", fact_type[index])
