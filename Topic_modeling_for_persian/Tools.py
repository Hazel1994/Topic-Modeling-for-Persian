import os
import re
from collections import Counter
import math
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from plot import plot_learning_curve
import matplotlib.pyplot as plt
import numpy as np


#read all the data from the given directory
def Read_data(folder):
    Data=[]
    labels_name=0
    #go over each file directory and read all the text
    for filename in os.listdir(folder):
        #add a class to each file
        labels_name += 1
        path1 = os.path.join(folder, filename)
        for i in os.listdir(path1):
            path2text = os.path.join(path1, i)
            #read the text file and add it to Data
            with open(path2text, "r",encoding="utf8") as myfile:
                sample = myfile.read()
                Data.append([ sample,labels_name])

    return Data

#tokenize the text
def TOK(text):
    #remove all the numbers
    text=re.sub("[0-9]", " ", text)
    #remove all english character
    text = re.sub("[a-zA-Z]", " ", text)
    #split the text to its words
    return text.split()

#extracting BOW features from the Data
def vectorizer(Data):

    Corpus = []
    dict = {}
    #add all the text to Coprus
    for i in Data:
        Corpus.append(i[0])
    #for each text we  tokenize it and create the vocabulary using all the words
    for i in Corpus:
        text = TOK(i)
        for j in text:
            #check if the word is in the vocabulary
            if j in dict:
                dict[j]+= 1
            else:
                dict[j] = 1
    #count the number of accurance of each word and take the 1000 most frequent words
    Vocabulary = Counter(dict).most_common(1000)
    #seperqate the words
    Vocabulary = [x[0] for x in Vocabulary]
    data_vectors = []
    #define a feature vectors for each document
    vector = np.zeros(1000, dtype=float)

    #doing the same thing for test data
    for i in Data:
        text = TOK(i[0])
        for j in text:
            if j in Vocabulary:
                vector[Vocabulary.index(j)] += 1
        vector = np.asanyarray([math.log(x + 1) for x in vector])
        data_vectors.append([vector, i[1]])
        vector = np.zeros(1000, dtype=float)
    #seperate the features and theit labels
    X,Y = np.asanyarray([i[0] for i in data_vectors]), np.asanyarray([i[1] for i in data_vectors])

    return X,Y

#spliting the data fairly so test and traing data have sample of all the classes
def Split_data(X,Y):
    X_test = []
    X_train = []
    y_train = []
    y_test = []

    #for each class we select the 2 sample to the test and the rest to the train data
    for i in range(7):
        #find class with lable equal to (i+1)
        indecis = np.where(Y == i + 1)
        test_index = indecis[0][0:2]
        train_index = indecis[0][2:]
        #append the test data
        for te in test_index:
            X_test.append(X[te, :])
            y_test.append(Y[te])
        # append the train data
        for tr in train_index:
            X_train.append(X[tr, :])
            y_train.append(Y[tr])

    return np.asanyarray(X_train),np.asanyarray(X_test),np.asanyarray(y_train),np.asanyarray(y_test)

#evaluate using MLP and naive bayes
def evaluate(X,Y,X_train, X_test, y_train, y_test):


    print("\n\nRunning neural network...")
    nn_clf = MLPClassifier()
    #define MLP and set learning rate to 0.01
    MLPClassifier(solver='adam', activation='tanh', early_stopping=True,learning_rate=.01, alpha=1e-5,
                hidden_layer_sizes=(300, 200))
    #traing MLP
    nn_clf.fit(X_train, y_train)
    #predict the test data
    y_pred = nn_clf.predict(X_test)
    print('accuracy on test data :',accuracy_score(y_test, y_pred))
    # predict the train data
    y_pred = nn_clf.predict(X_train)
    print("accuracy on train data : ",accuracy_score(y_train, y_pred))

    title = "Learning Curves (neural network)"

    #plot the learning curve
    plot_learning_curve(nn_clf, title, X, Y, ylim=(-0.1, 1.01), cv=5)
    plt.show()

    # ___________________ using naive bayes__________________#
    print("\n\nRunning Naive bayes...")
    NB_model = GaussianNB()
    NB_model = NB_model.fit(X=X_train, y=y_train)
    NB_model.fit(X_train, y_train)
    y_pred = NB_model.predict(X_test)
    print('accuracy on test data :', accuracy_score(y_test, y_pred))

    y_pred = NB_model.predict(X_train)
    print("accuracy on train data : ", accuracy_score(y_train, y_pred))

    title = "Learning Curves (Naive bayes)"

    plot_learning_curve(NB_model, title, X, Y, ylim=(-0.1, 1.01), cv=5)
    plt.show()

