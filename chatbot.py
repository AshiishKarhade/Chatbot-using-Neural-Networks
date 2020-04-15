# Libraries needed for Tensorflow processing
import tensorflow as tf
import numpy as np
import keras
import random
import json
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


with open('intents.json') as jsonfile:
    intents = json.load(jsonfile)

words = []
classes = []
documents = []
ignore = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(words)