#import necessary  libraries
from ADALINE import Adaline
import copy
from Tools import Read_data,vectorizer,Split_data,evaluate
from sklearn.metrics import accuracy_score

#dataset directory
folder='data'
#read the data using read_data function
Data=Read_data(folder)

#extracting features using vectorizer function
X,Y=vectorizer(Data)

# split the data so we will have 2 documents as test data for each category
X_train,X_test,y_train,y_test=Split_data(X,Y)


# print('running ADALINE classifer using OVA technique...\n')
#
# #make a copy of train labels
# y=copy.copy(y_train)
#
# #all the 7 classifers will be stored in this list
# classifiers=[]
# # for each class train a adaline classifer
# for i in range(7):
#     #create a Adaline classifer with larning rate = 0.0000001
#     ada = Adaline(eta=0.0000001)
#     #make a binay dataset
#     y[y!=i+1]=-1
#     y[y==i+1]=1
#     #train the classifer
#     ada.fit(X_train,y)
#     #predict the traning data
#     y_pred=ada.predict_int(X_train)
#     print('accuracy of classifer',str(i+1),' : ' ,accuracy_score(y,y_pred))
#     #assign th ogirinal dataset to y for the next loop
#     y = copy.copy(y_train)
#     #add the classifier to the list
#     classifiers.append(ada)
#
# #predict the test and the traing data using all th classifers
# predicted_labels_train=[]
# predicted_labels_test=[]
# for clf in classifiers:
#     predicted_labels_train.append(clf.predict(X_train))
#     predicted_labels_test.append(clf.predict(X_test))
#
# #using one versus all technique to classify each sample in train data
# final_classes_test=[]
# final_classes_train=[]
# for i in range(len(predicted_labels_train[0])):
#     col=[]
#     for j in range(7):
#         col.append(predicted_labels_train[j][i])
#         #assign the max probability of each class to the final labels
#     final_classes_train.append(col.index(max(col))+1)
#
# #using one versus all technique to classify each sample in test data
# for i in range(len(predicted_labels_test[0])):
#     col=[]
#     for j in range(7):
#         col.append(predicted_labels_test[j][i])
#     final_classes_test.append(col.index(max(col))+1)
#
# #display the final result using ADALINE
# print('final accuracy on train data: ', accuracy_score(final_classes_train,y_train))
# print('final accuracy on test data: ', accuracy_score(final_classes_test,y_test))

# running on MLP and Naive bayes
evaluate(X,Y,X_train, X_test, y_train, y_test)

