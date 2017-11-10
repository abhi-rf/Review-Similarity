# -*- coding: utf-8 -*-

import re
import nltk
import json
from nltk.corpus import stopwords
import ctypes, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
numpy.set_printoptions(threshold=numpy.nan)


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


#Fetching Reviews from JSON file
def fetch_reviews():
   # with open('C:/Users/Dell/Downloads/reviews_Electronics_5.json',encoding='utf-8') as data_file:
    #    return json.load(data_file.read())    

    #json_data = json.load(open('C:/Users/Dell/Downloads/reviews_Electronics_5.json'))
    json_data = json.load(open('C:/Users/Dell/Desktop/reviews.json'))
    return json_data

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stopword_set]

    # join the cleaned words in a list
    cleaned_word_list = " ".join(meaningful_words)
    
    stemmer = nltk.stem.snowball.SnowballStemmer("english",ignore_stopwords=True)
    
    cleaned_word_list = stemmer.stem(cleaned_word_list)

    return cleaned_word_list


def cosine_calculator(reviews):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    cs = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
    cs = numpy.array(cs)
    numpy.sort(cs)
    return cs
    

if __name__ == "__main__":
    reviews=[]
    reviews_fetched = fetch_reviews()
    #n is for the final acceptable reviews we will have after preprocessing and having more than 10 words
    n = 0 
    for i in range(len(reviews_fetched)):
        reviews_fetched[i]=preprocess((reviews_fetched[i]["reviewText"]))        
        if(len(reviews_fetched[i])>10):
            reviews.append(reviews_fetched[i])
            n = n+1
            
    for i in range(n):
        print(i," ",len(reviews[i]))

    
    #print(cs)
    for i in range(n):
        reviews.insert(0,reviews[i])
        cs = cosine_calculator(reviews)
        print(cs.shape)
        print(cs)
        reviews.pop(0)
        #for i in range(n):
    