import numpy as np
import pickle

with open("squad_retrieval_data.pickle", "rb") as handle:
    squad_retrieval_data = pickle.load(handle)

ques_train_embds = np.load("ques_train_embds.npy")
ques_dev_embds = np.load("ques_dev_embds.npy")
sents_embds = np.load("sents_embds.npy")


# TODO: look over the dev questions, see what is retrieved, and compute the MRR.



ques_train = squad_retrieval_data["train_list"]
ques_dev = squad_retrieval_data["dev_list"]
sent_list = squad_retrieval_data["sent_list"]
doc_list  = squad_retrieval_data["doc_list"]
resp_list = squad_retrieval_data["resp_list"]

def generate_mrr_from_embeddings():
    # This is commented out because it takes too much memory when evaluating the training set
    #scores = np.matmul(ques_train_embds, np.transpose(sents_embds))  # scores: row=different queries, col = different fact scores

    mrr = list([])
    for i, question_dict in enumerate(ques_train):
        question_score = np.matmul(ques_train_embds[i, :].reshape((1, 512)), np.transpose(sents_embds))[0,:]
        #question_score = scores[i,:]
        facts_ranked_by_scores = list(question_score.argsort())[::-1]
        gold_ranking = facts_ranked_by_scores.index(question_dict["response"])
        mrr.append(1/(1+gold_ranking))

        if (i+1)%200==0:
            print("processing %s out of %s", str(i+1), str(len(ques_train)))

        if i<20:
            print("="*20)
            print("\tquestion:", question_dict["question"])
            print("\tgold answer:", sent_list[question_dict["answer"]], sent_list[int(resp_list[question_dict["response"]][0])])
            print("\tgold ranking:", gold_ranking)
            print("\ttop 10 retrieved facts:")
            for j in range(10):
                print("\t\t", sent_list[int(resp_list[facts_ranked_by_scores[j]][0])])

            input("\twait for next example")

    print("mrr:", sum(mrr)/len(mrr))

    np.save("mrr_train_20200414.npy", np.array(mrr))

def check_generated_mrr():
    mrr = np.load("mrr_20200414.npy")

    histo, _ = np.histogram(mrr, bins = np.arange(0, 1.1, 0.1))
    print("histo of mrr:", histo)

    # for i, score in enumerate(mrr):

    return 0

generate_mrr_from_embeddings()
#check_generated_mrr()

