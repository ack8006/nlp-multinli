import nltk


def custom_tokenizer(x):
    return nltk.word_tokenize(x.lower())
