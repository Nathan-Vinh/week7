import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_lg
import string

def remove_punct(text):
    text_nopunct = "".join(
        [char for char in text if char not in string.punctuation])
    return text_nopunct.lower()

def toknize(text):
    nlp = spacy.load("en_core_web_lg")
    tokens = nlp(text)
    return tokens.text.split(" ")


def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in STOP_WORDS]
    return text


def lemmatizing(tokenized_text):
    nlp = spacy.load("en_core_web_lg")
    doc = str(tokenized_text).replace('[', '').replace(']', '')
    doc = nlp(doc)
    text = [token.lemma_ for token in doc]
    text = [w for w in text if w.isalpha() or w.isdigit()]
    return text


def tostring(text):
    doc = str(text).replace('[', '').replace(']', '')
    return doc
