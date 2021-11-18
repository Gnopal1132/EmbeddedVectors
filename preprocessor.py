# Import Libraries and Setting things up
import pandas as pd
import numpy as np
import pickle
import unicodedata
import torch
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import sys,nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.append('list')
stop_words = set(stop_words)
from collections import Counter

import re

import string

from nltk.stem import PorterStemmer


# A Helper Method

stemmer = PorterStemmer()


def join_bigrams(text):
    text = text.replace('artificial intelligence', 'artificialintelligence')
    text = text.replace('machine learning', 'machinelearning')
    text = text.replace('deep learning', 'deeplearning')
    text = text.replace('data science', 'datascience')
    text = text.replace('social media', 'socialmedia')

    return text


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def remove_urls(text):
    return re.sub(r'http\S+', '', text)


def extract_hashtags(text):
    # initializing hashtag_list variable
    hashtag_list = []

    # splitting the text into words
    for word in text.split():

        # checking the first charcter of every word
        if word[0] == '#':
            # adding the word to the hashtag_list
            hashtag_list.append(word[1:])

    return hashtag_list

def extract_mentions(text):
    # initializing hashtag_list variable
    hashtag_list = []

    # splitting the text into words
    for word in text.split():

        # checking the first charcter of every word
        if word[0] == '@':
            # adding the word to the hashtag_list
            hashtag_list.append(word[1:])

    return hashtag_list

def remove_rt(text):
    if text[:2] == 'RT':
        try:
            # text = text.split(':', 1)[1]
            text = ' '.join(text.split(' ')[2:])
        except:
            print(text)
            raise NotImplementedError
        return text
    else:
        return text

def clean_string(text, tokens=True):
    text = remove_rt(text)
    text = text.lower().strip()

    text = remove_urls(text)
    text = remove_accents(text)
    text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    # text = expand_bigrams(text)
    # text = my_translator.translate(text)
    if tokens:
        text = join_bigrams(text)
        text = text.split()
        text = [stemmer.stem(t.strip()) for t in text if not t in stop_words]
        return text
    else:
        return text



def compute_topics_from_tokens(tokens_list, n=5):
    flat_list = []
    for sublist in tokens_list:
        if isinstance(sublist, list): flat_list.extend(sublist)

    counts = Counter(flat_list)
    topics = counts.most_common(n)

    return topics


# A class for generating the Corpus and Vocabulary

class CorpusExtraction:
    def __init__(self, result_api):
        self.result = result_api

    def return_by_user(self, frame):
        text_by_user = {}
        
        # Lets first deal with user in tweet_user_id_str
        unique_id_tweet = set(frame['tweet_user_id_str'])
        for user in list(unique_id_tweet):
            if user is None:
                continue
            temp = frame.loc[frame['tweet_user_id_str'] == user,:]
            user_txt_retweet = list(temp.loc[temp['type'] != 'retweet','text'])
            if len(user_txt_retweet) > 0:
                text_by_user[user] = user_txt_retweet
        
        
        # Lets now deal with guys in the interaction
        unique_id_interacted = set(frame['tweet_user_id_str_interaction'])
        for user in list(unique_id_interacted):
        
            if user is None:
                continue
            
            user_txt = list(frame.loc[(frame['tweet_user_id_str_interaction'] == user)&(frame['text_interaction'].notnull()), 'text_interaction'])
            if len(user_txt) > 0:
                if user in text_by_user:
                    text_by_user[user].extend(user_txt)
                else:
                    text_by_user[user] = user_txt
        
        
        for user,user_txt in text_by_user.items():
            text_by_user[user] = clean_string(' '.join(user_txt))
        
        return text_by_user


    def _get_bow_for_user(self, result,ordered=False):
        BOW = OrderedDict() if ordered else {}
        for user,value in result.items():
            BOW[user] = dict(FreqDist(value))
        return BOW


    def return_text(self, BOW):
        final_string = ''
        for word, times in BOW.items():
            final_string += (word+' ')*times
        return final_string.strip()
    

    def document_matrix(self, bow, vocab_size, create_vocab=False, threshold=0.1):
        full_document = []
        for key, string in bow.items():
            text = self.return_text(string)
            full_document.append(text)

        if create_vocab:
            vectorizer = TfidfVectorizer(max_features=vocab_size, max_df=threshold)
            vectorizer.fit(full_document)
            save_vectorizer(vectorizer=vectorizer, write=True)
        else:
            vectorizer = save_vectorizer(write=False)

        features = vectorizer.get_feature_names()
        vectorizer = CountVectorizer(vocabulary=features)
        X = vectorizer.fit_transform(full_document)

        final_frame = pd.DataFrame(X.toarray())
        final_frame.columns = features
        temp = final_frame.sum(axis=1) > 0
        proportion = temp.mean()
        return final_frame, 1. - proportion
    

    def return_corpus_with_proportion(self, vocab_size=100, create_vocab=False,threshold = 0.1):
        cleaned_text_by_user = self.return_by_user(self.result)
        
        BOW = self._get_bow_for_user(cleaned_text_by_user)
        corpus, proportion = self.document_matrix(BOW, vocab_size, create_vocab = create_vocab,
                                                 threshold=threshold) 
        Vocabulary = list(corpus.columns)
        return Vocabulary,corpus,proportion



# A Helper function for Generating the Train and Validation Split

def give_train_val_set(dataset, train_size=0.8):
  np.random.shuffle(dataset)
  train_size = int(len(dataset)*train_size)
  X_train = dataset[:train_size]
  X_valid = dataset[train_size:]
  return X_train, X_valid

# A Helper Function to Save the Dataloaders as Pickle Files
def save_me_loader(loader,is_train=True, Vocab=False):
    if not Vocab:
        file = open('train.pickle', 'wb') if is_train else open('val.pickle', 'wb')
    else:
        file = open('vocab.pickle', 'wb')
    
    pickle.dump(loader, file)
    file.close()
    
    if Vocab:
        print('Vocab Saved!')
        return
    
    if is_train:
        print('Train data Generated as Pickle!!')
    else:
        print('Val data Generated as Pickle!!')
    return


# To save the Tfid Vectorizer
def save_vectorizer(vectorizer=None, write=True):
    file = open('vectorizer.pickle', 'wb') if write else open('vectorizer.pickle', 'rb')
    
    if write:
        pickle.dump(vectorizer, file)
        file.close()
    else:
        vectorizer = pickle.load(file)
        file.close()
        return vectorizer


def Preprocessing(Corpus):
    # Lets first remove those entries where everything is False
    condition = np.any(Corpus, axis=1) == False
    index = np.where(condition)[0]
    modified_corpus = np.delete(Corpus, index, axis = 0)

    # Lets Normalize
    sum = np.sum(modified_corpus, axis=1, keepdims=True)
    modified_corpus = modified_corpus / sum

    return modified_corpus


if __name__ == '__main__':
    # Load The Pickle File
    file = open('population.pickle', 'rb')
    result = pickle.load(file)
    file.close()

    # Main Function Calls
    corpus_generator = CorpusExtraction(result)
    vocab_size = 1000
    threshold = 0.70

    # Corpus is a DataFrame
    Vocab, Corpus, Proportion = corpus_generator.return_corpus_with_proportion(vocab_size=vocab_size, create_vocab=True,
                                                                               threshold=threshold)
    preprocessed_corpus = Preprocessing(Corpus.values)
    train, valid = give_train_val_set(preprocessed_corpus)

    # Generating the DataLoader

    save_me_loader(train)
    save_me_loader(valid, is_train=False)
    save_me_loader(Vocab, Vocab=True)





