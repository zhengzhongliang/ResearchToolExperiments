import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

questions = ["What is your age?"]
responses = ["I am 20 years old.", "good morning"]
response_contexts = ["I will be 21 next year.", "great day."]

module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

question_embeddings = module.signatures['question_encoder'](
            tf.constant(questions))
response_embeddings = module.signatures['response_encoder'](
        input=tf.constant(responses),
        context=tf.constant(response_contexts))

np.inner(question_embeddings['outputs'], response_embeddings['outputs'])

