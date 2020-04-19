#@title Setup Environment
# Install the latest Tensorflow version.
# !pip install tensorflow_text
# !pip install simpleneighbors
# !pip install nltk

import json
import nltk
import os
import pprint
import random
import simpleneighbors
import urllib
from IPython.display import HTML, display

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import numpy as np
from google.colab import output
from google.colab import files
import pickle

nltk.download('punkt')


def download_squad(url):
  return json.load(urllib.request.urlopen(url))

def extract_sentences_from_squad_json(squad):
  all_sentences = []
  for data in squad['data']:
    for paragraph in data['paragraphs']:
      sentences = nltk.tokenize.sent_tokenize(paragraph['context'])
      all_sentences.extend(zip(sentences, [paragraph['context']] * len(sentences)))
  return list(set(all_sentences)) # remove duplicates

def extract_questions_from_squad_json(squad):
  questions = []
  for data in squad['data']:
    for paragraph in data['paragraphs']:
      for qas in paragraph['qas']:
        if qas['answers']:
          questions.append((qas['question'], qas['answers'][0]['text']))
  return list(set(questions))

def generate_query_embeddings(model, instances_list):
    batch_size = 100

    embeddings_list = list()

    print('Computing embeddings for %s questions' % len(instances_list))
    slices = zip(*(iter(instances_list),) * batch_size)
    num_batches = int(len(instances_list) / batch_size)
    for n, s in enumerate(slices):
        output.clear(output_tags='progress')
        with output.use_tags('progress'):
            print('Processing batch %s of %s' % (n + 1, num_batches))

        question_batch = list([question_dict["question"] for question_dict in s])
        encodings = model.signatures['question_encoder'](tf.constant(question_batch))
        for i in range(len(question_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    if num_batches*batch_size<len(instances_list):
        question_batch = list([question_dict["question"] for question_dict in instances_list[num_batches*batch_size:]])
        encodings = model.signatures['question_encoder'](tf.constant(question_batch))
        for i in range(len(question_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    return np.array(embeddings_list)

def generate_document_embeddings(model, response_list, sent_list, doc_list):
    batch_size = 100

    embeddings_list = list()

    print('Computing embeddings for %s questions' % len(response_list))
    slices = zip(*(iter(response_list),) * batch_size)
    num_batches = int(len(response_list) / batch_size)
    for n, s in enumerate(slices):
        output.clear(output_tags='progress')
        with output.use_tags('progress'):
            print('Processing batch %s of %s' % (n + 1, num_batches))

        response_batch = list([sent_list[int(sent_id)] for sent_id, doc_id in s])
        context_batch = list([doc_list[int(doc_id)] for sent_id, doc_id in s])
        encodings = model.signatures['response_encoder'](
            input=tf.constant(response_batch),
            context=tf.constant(context_batch)
        )
        for i in range(len(response_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    if batch_size*num_batches<len(response_list):
        response_batch = list([sent_list[int(sent_id)] for sent_id, doc_id in response_list[num_batches*batch_size:]])
        context_batch = list([doc_list[int(doc_id)] for sent_id, doc_id in response_list[num_batches*batch_size:]])
        encodings = model.signatures['response_encoder'](
            input=tf.constant(response_batch),
            context=tf.constant(context_batch)
        )
        for i in range(len(response_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))


    return np.array(embeddings_list)


def main():
    #@title Load model from tensorflow hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3", "https://tfhub.dev/google/universal-sentence-encoder-qa/3"]
    model = hub.load(module_url)

    with open("squad_retrieval_data.pickle", "rb") as handle:
        squad_retrieval_data = pickle.load(handle)

    questions_train_embds =generate_query_embeddings(model, squad_retrieval_data["train_list"])
    questions_dev_embds =generate_query_embeddings(model, squad_retrieval_data["dev_list"])
    sentences_embds =generate_document_embeddings(model, squad_retrieval_data["resp_list"], squad_retrieval_data["sent_list"], squad_retrieval_data["doc_list"])

    print(questions_train_embds.shape)
    print(questions_dev_embds.shape)
    print(sentences_embds.shape)

    np.save("ques_train_embds.npy", questions_train_embds)
    np.save("ques_dev_embds.npy", questions_dev_embds)
    np.save("sents_embds.npy", sentences_embds)

    files.download("ques_train_embds.npy")
    files.download("ques_dev_embds.npy")
    files.download("sents_embds.npy")

# def interpret_embd():
#     with open("jev-v1.1.json", "r") as handle:
#         sentences_dev_json = json.load(handle.read())
#
#     print(sentences_dev_json.keys())
#
#
#     return 0
#
# interpret_embd()

