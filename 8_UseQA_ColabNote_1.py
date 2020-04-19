#@title Setup Environment
# Install the latest Tensorflow version.
# !pip install tensorflow_text
# !pip install simpleneighbors
# !pip install nltk

import json
# import nltk
# import os
# import pprint
# import random
# import simpleneighbors
# import urllib
# from IPython.display import HTML, display
#
# import tensorflow.compat.v2 as tf
# import tensorflow_hub as hub
# from tensorflow_text import SentencepieceTokenizer
# import numpy as np
# from google.colab import output
# from google.colab import files

# nltk.download('punkt')


# def download_squad(url):
#   return json.load(urllib.request.urlopen(url))
#
# def extract_sentences_from_squad_json(squad):
#   all_sentences = []
#   for data in squad['data']:
#     for paragraph in data['paragraphs']:
#       sentences = nltk.tokenize.sent_tokenize(paragraph['context'])
#       all_sentences.extend(zip(sentences, [paragraph['context']] * len(sentences)))
#   return list(set(all_sentences)) # remove duplicates
#
# def extract_questions_from_squad_json(squad):
#   questions = []
#   for data in squad['data']:
#     for paragraph in data['paragraphs']:
#       for qas in paragraph['qas']:
#         if qas['answers']:
#           questions.append((qas['question'], qas['answers'][0]['text']))
#   return list(set(questions))
#
# def generate_embeddings(model, text_file_list, encoder_type):
#     batch_size = 100
#
#     embeddings_list = list()
#
#     print('Computing embeddings for %s sentences' % len(text_file_list))
#     slices = zip(*(iter(text_file_list),) * batch_size)
#     num_batches = int(len(text_file_list) / batch_size)
#     for n, s in enumerate(slices):
#         output.clear(output_tags='progress')
#         with output.use_tags('progress'):
#             print('Processing batch %s of %s' % (n + 1, num_batches))
#
#         if encoder_type=="question_encoder":
#             question_batch = list([q for q, a in s])
#             encodings = model.signatures['question_encoder'](tf.constant(question_batch))
#             for i in range(len(question_batch)):
#                 embeddings_list.append(np.array(encodings['outputs'][i]))
#         else:
#             response_batch = list([r for r, c in s])
#             context_batch = list([c for r, c in s])
#             encodings = model.signatures['response_encoder'](
#                 input=tf.constant(response_batch),
#                 context=tf.constant(context_batch)
#               )
#             for i in range(len(response_batch)):
#                 embeddings_list.append(np.array(encodings['outputs'][i]))
#
#     return np.array(embeddings_list)
#
#
# def main():
#     #@title Load model from tensorflow hub
#     module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3", "https://tfhub.dev/google/universal-sentence-encoder-qa/3"]
#     model = hub.load(module_url)
#
#     squad_url_dev = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json' #@param ["https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"]
#
#     squad_json_dev = download_squad(squad_url_dev)
#     sentences_dev = extract_sentences_from_squad_json(squad_json_dev)
#     questions_dev = extract_questions_from_squad_json(squad_json_dev)
#
#     questions_dev_embds =generate_embeddings(model, text_file_list=questions_dev, encoder_type="question_encoder")
#     sentences_dev_embds =generate_embeddings(model, text_file_list=sentences_dev, encoder_type="response_encoder")
#
#     print(questions_dev_embds.shape)
#     print(sentences_dev_embds.shape)
#
#     np.save("ques_dev_embds.npy", questions_dev_embds)
#     np.save("sents_dev_embds.npy", sentences_dev_embds)
#
#     files.download("ques_dev_embds.npy")
#     files.download("sents_dev_embds.npy")

def interpret_embd():
    with open("jev-v1.1.json", "r") as handle:
        sentences_dev_json = json.load(handle.read())

    print(sentences_dev_json.keys())


    return 0

interpret_embd()

