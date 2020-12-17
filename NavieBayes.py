import pickle
from utils import process_tweet, lookup
import pdb
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd
from tkinter import *

nltk.data.path.append("C:/Users/saurabh yadav/AppData/Roaming/nltk_data/corpora")


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg


train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

#---------------------------------------------------------------------------

def count_tweets(result, tweets, ys):
    
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word,y)
            
            if pair in result:
                
                result[pair] += 1
                
            else:
                
                result[pair] = 1
                
    return result


#---------------------------------------------------------------------------

freqs = count_tweets({}, train_x, train_y)

#---------------------------------------------------------------------------

def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    N_pos = N_neg = V_pos = V_neg = 0
    
    for pair in freqs.keys():
        
        if pair[1] > 0:
            
            V_pos += 1
            N_pos += freqs[pair]
            
        else:
            
            V_neg += 1
            N_neg += freqs[pair]
            
    D = len(train_y)
    D_pos = (len(list(filter(lambda x: x > 0, train_y))))
    D_neg = (len(list(filter(lambda x: x <= 0, train_y))))
    logprior = np.log(D_pos) - np.log(D_neg)
    
    for word in vocab:
        
        freq_pos = lookup(freqs,word,1)
        freq_neg = lookup(freqs,word,0)
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)

    return logprior, loglikelihood

#---------------------------------------------------------------------------

logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)

#---------------------------------------------------------------------------

def naive_bayes_predict(tweet, logprior, loglikelihood):
    word_l = process_tweet(tweet)
    p = 0
    p += logprior
    
    for word in word_l:
        
        if word in loglikelihood:
            
            p += loglikelihood[word]

    return p

#---------------------------------------------------------------------------

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0  
    y_hats = []
    
    for tweet in test_x:
        
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            
            y_hat_i = 1
            
        else:
            
            y_hat_i = 0
            
        y_hats.append(y_hat_i)

    error = np.mean(np.absolute(y_hats-test_y))

    accuracy = 1-error

    return accuracy

#---------------------------------------------------------------------------


def get_ratio(freqs, word):
    
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    pos_neg_ratio['positive'] = lookup(freqs,word,1)
    pos_neg_ratio['negative'] = lookup(freqs,word,0)
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1)/(pos_neg_ratio['negative'] + 1)
  
    return pos_neg_ratio


#---------------------------------------------------------------------------

get_ratio(freqs, 'happi')

#---------------------------------------------------------------------------

def get_words_by_threshold(freqs, label, threshold):
    
    word_list = {}
    
    for key in freqs.keys():
        
        word, _ = key
        pos_neg_ratio = get_ratio(freqs, word)
        
        if label == 1 and pos_neg_ratio['ratio'] >= threshold :
            
            word_list[word] = pos_neg_ratio
            
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
            
            word_list[word] = pos_neg_ratio

    return word_list

#-----------------------------------------------------------------------

def submit_fields():

    path = 'sentiment.xlsx'
    df1 = pd.read_excel(path)
    SeriesA = df1['Tweet']
    SeriesB = df1['Sentiment']
    SeriesC = df1['Value']
    A = pd.Series(entry1.get())
    my_tweet = entry1.get()
    p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
    
    if (p>0):
        
        B = pd.Series("Possitive")
        C = pd.Series(p)
        
    elif(p==0):
        
        B = pd.Series("Netural")
        C = pd.Series(p)
        
    else:
        
        B = pd.Series("Negative")
        C = pd.Series(p)
        
    SeriesA = SeriesA.append(A)
    SeriesB = SeriesB.append(B)
    SeriesC = SeriesC.append(C)
    df2 = pd.DataFrame({"Tweet":SeriesA, "Sentiment":SeriesB,"Value":SeriesC})
    df2.to_excel(path, index=False)


#---------------------------------------------------------------------------


def show_entry_fields():

    my_tweet = entry1.get()
    p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
    print("Your Comment :",my_tweet)

    if (p>0):
        print("Possitive")
        print("Value :",p)
        print(" ")
    elif(p==0):
        print("Netural")
        print("Value :",p)
        print(" ")
    else:
        print("Negative")
        print("Value :",p)
        print(" ")
        

#---------------------------------------------------------------------------


def erase():
    entry1.delete(0, END)

master = Tk()
master.title("Sentiment Analyser")
master.geometry("350x200")

Label(master, text="Tweet").grid(row=0)


entry1 = Entry(master)
entry1.grid(row=0, column=1, ipadx=50)


Button(master, text='Quit', command=master.quit).grid(row=3,column=0, pady=4)

Button(master, text='Analyse', command=show_entry_fields).grid(row=3,column=1, pady=4)

Button(master, text='Submit', command=submit_fields).grid(row=4,column=0, pady=4)

Button(master, text='Erase', command=erase).grid(row=4,column=1, pady=4)

mainloop()