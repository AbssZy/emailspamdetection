#creator : AbssZy

import nltk
import random
import os
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
import string
from warnings import simplefilter
simplefilter(action='ignore',category=FutureWarning)
from warnings import simplefilter
simplefilter(action='ignore',category=FutureWarning)

def pr(word,msg):
    if word in msg:
        return 1
    else:
        return 0

def find_feature(word_features,message):
    feature = {}
    for word in word_features:
        feature[word] = pr(word,message)
    return feature

def create_mnb_classifier(trainingset,testingset):
    x=0
    y=0
    print("\nMultinomial Naive Bayes classifier is being trained and created....")
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(trainingset)
    for t in testingset:
        y=y+1
        l=MNB_classifier.classify(t[0])
        if(l==t[1]):
            x=x+1
    accuracy=x/y * 100
    print("Multinomial Classifier accuracy = "+ str(accuracy))
    return MNB_classifier

def create_bnb_classifier(trainingset,testingset):
    x=0
    y=0
    print("\nBernoulli Naive Bayes classifier is being trained and created...")
    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(trainingset)
    for t in testingset:
        y=y+1
        l=BNB_classifier.classify(t[0])
        if(l==t[1]):
            x=x+1
    accuracy=x/y * 100
    print("BernoulliNB accuracy precent = " + str(accuracy))
    return BNB_classifier

def create_logistic_regression_classifier(trainingset,testingset):
    x=0
    y=0
    print("\nLogistic Regreesion classifier is being trained and created...")
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(trainingset)
    for t in testingset:
        y=y+1
        l=LogisticRegression_classifier.classify(t[0])
        if(l==t[1]):
            x=x+1
        accuracy=x/y * 100
        print("Logistic Regression classifier accuracy = " + str(accuracy))
        return LogisticRegression_classifier

def create_training_testing():
    with open("SMSSpamCollection.txt") as f:
        messages = f.read().split('\n')
    print("Creating bag of words. ")
    all_message = []
    all_words = []
    for message in messages:
        if message.split('\t')[0] == "spam":
            all_message.append([message.split('\t'), "spam"])
        else:
            all_message.append([message.split('\t'), "ham"])
        for s in string.punctuation:
            if s in message:
                message = message.replace(s, " ")
        stop = stopwords.words('english')
        for word in message.split(" "):
            if not word in stop:
                all_words.append(word.lower())
    print("Bag of words created.")
    
    random.shuffle(all_message)
    random.shuffle(all_message)
    random.shuffle(all_message)
    
    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:2000]
    
    print("\nCreating feature set. ")
    featureset = [(find_feature(word_features,message),category) for (message,category) in all_message]
    print("Feature set created.")
    trainingset = featureset[:int(len(featureset)*3/4)]
    testingset = featureset[int(len(featureset)*3/4):]
    
    print("\nLength of feature set ",len(featureset))
    print("Length of training set",len(trainingset))
    print("Length of testing set",len(testingset))
    
    return word_features, featureset, trainingset, testingset

def main():
    word_features, featureset, trainingset, testingset = create_training_testing()
    MNB_classifier = create_mnb_classifier(trainingset, testingset)
    BNB_classifier = create_bnb_classifier(trainingset, testingset)
    LR_classifier = create_logistic_regression_classifier(trainingset, testingset)
    mail = input('enter message:').lower()
    x=0
    print("\n")
    print("Multinomial Naive Bayes")
    print(" ")
    feature = find_feature(word_features,mail)
    print(MNB_classifier.classify(feature))
    if(MNB_classifier.classify(feature)=="ham"):
        x=x+1
    print("\n")
    print("Bernoulli Naive Bayes")
    print(" ")
    feature = find_feature(word_features, mail)
    print(BNB_classifier.classify(feature))
    if(BNB_classifier.classify(feature)=="ham"):
        x=x+1
    print("\n")
    print("Logistic Regression")
    print(" ")
    feature = find_feature(word_features,mail)
    print(LR_classifier.classify(feature))
    if(LR_classifier.classify(feature)=="ham"):
        x=x+1
    if(x>=2):
        print("\n*******\n message is classified as ham\n*******\n")
    else:
        print("\n*******\n message is classified as spam\n*******\n")
main()
