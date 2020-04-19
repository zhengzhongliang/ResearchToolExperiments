import numpy as np
import pickle
import random
from sklearn.linear_model import LogisticRegression


def build_lr_dataset(tfidf_result_list, useqa_mrr_list, train_num):
    assert(train_num<len(tfidf_result_list))

    # generate train input and labels:
    train_indices = list(range(train_num))
    random.shuffle(train_indices)

    features_train = np.zeros((train_num, 7))
    labels_train = [0] * train_num
    for n_sample, n_original_index in enumerate(train_indices):
        labels_train[n_sample] = 1 if tfidf_result_list[n_original_index]["gold_resp_mrr"]<useqa_mrr_list[n_original_index] else 0
        for n_feature in range(7):
            features_train[n_sample, n_feature] = np.mean(tfidf_result_list[n_original_index]["top_scores"][:2**n_feature])

    # generate test input and labels:
    test_indices = list(range(train_num, len(tfidf_result_list)))

    features_test = np.zeros((len(test_indices), 7))
    labels_test = [0]*len(test_indices)
    for n_sample, n_original_index in enumerate(test_indices):
        labels_test[n_sample] = 1 if tfidf_result_list[n_original_index]["gold_resp_mrr"]<useqa_mrr_list[n_original_index] else 0
        for n_feature in range(7):
            features_test[n_sample, n_feature] = np.mean(tfidf_result_list[n_original_index]["top_scores"][:2**n_feature])

    test_mrr_tfidf = [tfidf_result_list[test_sample_index]["gold_resp_mrr"] for test_sample_index in test_indices]
    test_mrr_useqa = [useqa_mrr_list[test_sample_index] for test_sample_index in test_indices]

    return features_train, labels_train, features_test, labels_test, test_mrr_tfidf, test_mrr_useqa


def train_eval_logistic_regression(features_train, labels_train, features_test, labels_test, test_mrr_tfidf, test_mrr_useqa):
    print("evalualte using LR")
    mrr_list_to_return = list([])
    mrr_list_all_seeds = list([])
    n_bert_used = 0

    reg = LogisticRegression()

    reg.fit(features_train, labels_train)

    predictions = reg.predict(features_test)

    hybrid_mrr = []
    n_useqa_selected = 0
    for i, pred in enumerate(list(predictions)):
        if pred>0.5:
            n_useqa_selected+=1
            hybrid_mrr.append(test_mrr_useqa[i])
        else:
            hybrid_mrr.append(test_mrr_tfidf[i])

    print("test mrr tfidf:", sum(test_mrr_tfidf)/len(test_mrr_tfidf))
    print("test mrr useqa:", sum(test_mrr_useqa) / len(test_mrr_useqa))
    print("test mrr hybrid:", sum(hybrid_mrr) / len(hybrid_mrr))
    print("n useqa selected:", n_useqa_selected)
    print("reg coef:", reg.coef_)
    print("reg intercept:", reg.intercept_)


    return 0


def main():
    with open("squad_dev_result_tfidf.pickle", "rb") as handle:
        tfidf_result_list = pickle.load(handle)

    useqa_mrr_list = list(np.load("mrr_dev_20200414.npy"))

    features_train, labels_train, features_test, labels_test, test_mrr_tfidf, test_mrr_useqa = build_lr_dataset(tfidf_result_list, useqa_mrr_list, 2000)

    train_eval_logistic_regression(features_train, labels_train, features_test, labels_test, test_mrr_tfidf,test_mrr_useqa)

main()



