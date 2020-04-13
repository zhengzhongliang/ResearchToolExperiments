import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tf_sentencepiece


'''
References:
Universal sentence encoder: https://arxiv.org/pdf/1803.11175.pdf
Sentence encoder for QA: https://arxiv.org/pdf/1907.04307.pdf
'''
embedding_modules = {"universal_sentence_encoder": "https://tfhub.dev/google/universal-sentence-encoder/1", "sentence_encoder_for_qa":"https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1"}

def forward_unversal_sentence_encoder(module, text_batch:list):
    embeddings = module(text_batch)

    return embeddings

def forward_sentence_encoder_qa(module, text_batch:dict, encoder_type):
    assert (encoder_type in ["question_encoder", "response_encoder"], "encoder must be question_encoder or response_encoder")

    if encoder_type == "question_encoder":
        embeddings = module(
        dict(text=text_batch["query_text"]),
        signature="question_encoder", as_dict=True)

    else:
        embeddings = module(
        dict(text=text_batch["answer_text"],
             context=text_batch["context_text"]),
        signature="response_encoder", as_dict=True)

    return embeddings




def test_qa_encoder_simple():
    #embedding_module = hub.KerasLayer(embedding_modules["sentence_encoder_for_qa"])
    embedding_module = hub.Module(embedding_modules["sentence_encoder_for_qa"])


    text_batch = {}
    text_batch["query_text"] = ["Where are my shoes?", "What is the name of the president of US?", "How is the weather today?"]

    embeddings = forward_sentence_encoder_qa(embedding_module, text_batch, "question_encoder")

    embeddings = np.array(embeddings)

    print(embeddings.shape)

    return 0

def main():
    test_qa_encoder_simple()

    return 0

main()


