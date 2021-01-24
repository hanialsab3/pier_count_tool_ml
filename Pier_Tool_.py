import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, naive_bayes
from sklearn import model_selection, naive_bayes, svm
import nltk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,classification_report, confusion_matrix

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')


# Use the application default credentials
Corpus = pd.read_csv(r"C:\Users\hania\Documents\GitHub\pier_count_tool_ml\Pier_Count_Tool_Dataset_01.23.2021.csv",encoding='utf-8',low_memory=False)
Corpus=Corpus.replace(np.nan,"0")
Corpus.info()
print(Corpus.head())
TargetDF= Corpus[['EXT - Total Piers', 'EXT - Drive Piers',	'EDG - Total Piers', 'EDG - Drive Piers']]
print(TargetDF)

# Corpus.drop([0:3])
Corpus.drop(columns=['Project #'])
Corpus.drop(['Project #', 'Opportunity Title', 'Pier Load Calcs'], axis='columns',inplace=True)
print(Corpus.head())
Corpus.info()
# Corpus=Corpus.sample(frac=1)
# Corpus=Corpus.astype(float)
# TargetDF=TargetDF.astype(float)
# Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus, TargetDF ,test_size=0.2)

# # fit the training dataset on the NB classifier
# Naive = naive_bayes.MultinomialNB()
# SVM= svm.SVC(gamma='scale')
# Naive.fit(Train_X,Train_Y)
# SVM.fit(Train_X,Train_Y)
#
# # predict the labels on validation dataset
# predictions_NB = Naive.predict(Test_X)
#
# predictions_SVM = SVM.predict(Test_X)
#
# # Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score:",accuracy_score(predictions_NB, Test_Y)*100)
#
# print("SVM Accuracy Score:", accuracy_score(predictions_SVM, Test_Y)*100)
#
#
# print("f1 score:", f1_score(predictions_NB, Test_Y))
# print("precision score:",precision_score(predictions_NB, Test_Y))
# print("recall score score:",recall_score(predictions_NB,Test_Y))
# print("confusion matrix:",confusion_matrix(predictions_NB,Test_Y))
