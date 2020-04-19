import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # used for compute cosine similarity for sparse matrix
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower())]

def get_response_list_text(resp_tuples_list, sent_list, doc_list):
    resp_list_text = list()
    for (sent_id, doc_id) in resp_tuples_list:
        resp_list_text.append(sent_list[int(sent_id)]+" "+doc_list[int(doc_id)])

    return resp_list_text

def eval_tfidf(instances_list, tfidf_vectorizer, doc_matrix, kb,  saved_file_name):

    # What to save right now:
    # (1) top 64 retrieved response id
    # (2) top 64 retrieved response tf-idf score
    # (3) gold response mrr
    # (4) we may want to save all scores later, but not now.

    correct_count = 0
    justification_hit_ratio = list([])
    mrr = []
    list_to_save = list([])
    count_top10 = 0
    result_list_to_save = []
    for i, instance in enumerate(instances_list):
        query = [instance["question"]]
        query_matrix = tfidf_vectorizer.transform(query)

        cosine_similarities = linear_kernel(query_matrix, doc_matrix).flatten()
        rankings = list(reversed(np.argsort(cosine_similarities).tolist()))  # rankings of facts, from most relevant

        top_facts = rankings[:64]
        top_scores = cosine_similarities[top_facts]
        gold_resp_mrr = 1/(1+rankings.index(instance["response"]))

        result_list_to_save.append({"top_facts":top_facts, "top_scores":top_scores, "gold_resp_mrr": gold_resp_mrr})

        mrr.append(gold_resp_mrr)

        if i%10==0:
            print("processing example ",i)

    with open("squad_dev_result_tfidf.pickle", "wb") as handle:
        pickle.dump(result_list_to_save, handle)

    print(result_list_to_save[0])


    print("="*20)
    print("tfidf mrr:", sum(mrr)/len(mrr))

    return 0

def squad_retrieval_get_mrr():

    with open("squad_retrieval_data.pickle", "rb") as handle:
        squad_retrieval_data = pickle.load(handle)

    ques_dev = squad_retrieval_data["dev_list"]
    sent_list = squad_retrieval_data["sent_list"]
    doc_list = squad_retrieval_data["doc_list"]
    resp_list = squad_retrieval_data["resp_list"]

    resp_list_text = get_response_list_text(resp_list, sent_list, doc_list)

    stop_words_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                       "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                       "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
                       "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                       "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
                       "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                       "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
                       "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
                       "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
                       "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                       "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list)  #, tokenizer=LemmaTokenizer()  # even if we don't use any tokenizer, this still works.
    doc_matrix = tfidf_vectorizer.fit_transform(
        resp_list_text)

    eval_tfidf(ques_dev, tfidf_vectorizer, doc_matrix, resp_list_text, "")

    return 0

squad_retrieval_get_mrr()