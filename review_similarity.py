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
    json_data = json.load(open('C:/Users/Dell/Desktop/reviews_short.json'))
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
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews[0] for reviews in reviews)
    cs = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
    cs = numpy.array(cs)
    #numpy.sort(cs)
    return cs
    

if __name__ == "__main__":
    reviews=[]
    reviews_fetched = fetch_reviews()
    #n is for the final acceptable reviews we will have after preprocessing and having more than 10 words
    n = 0 
    for i in range(len(reviews_fetched)):
        reviews_fetched[i][0]=preprocess((reviews_fetched[i]["reviewText"]))
        reviews_fetched[i][1]=reviews_fetched[i]["reviewerID"]
        if(len(reviews_fetched[i][0])>10):
            reviews.append(reviews_fetched[i])
            n = n+1
            
    rev_id = []
    for i in range(n):
        #print(i," ",len(reviews[i][0]))
        rev_id.append(reviews[i][1])
    
    rev_id = numpy.asarray(rev_id)
    rev_id = rev_id.transpose()
    #print(rev_id.shape)
    similarity_matrix = []
    #print(cs)
    for i in range(n):
        reviews.insert(0,reviews[i])
        cs = cosine_calculator(reviews)
        
        #sorted(cs,key=lambda x: x[0])
        #cs = cs[numpy.argsort(cs[0,:])]
        #cs = cs[0].sort()
        cs = cs.transpose()
        #cs = numpy.insert(cs,1,values=reviews[][])
        #print(cs.shape)
        #print(cs)
        print(" ")
        reviews.pop(0)
        cs=numpy.delete(cs,0)
        #print(cs.shape)
        cs = numpy.column_stack((cs,rev_id))
        cs = cs[cs[:,0].argsort()[::-1]]
        cs = cs[0:4,:]
        print(cs)
        similarity_matrix.append(cs)
        #for i in range(n):
    
    