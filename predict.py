import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


nltk.download('punkt')

data = pd.read_table("smsspamcollection/SMSSpamCollection",

    sep="\t",
    header=None,
    names=["label","messages"]    
)

print(data.head())

# EDA
print("Shape of the date:",data.shape)
print("Non of null values:",data.isnull().sum().sum())
print("Info:",data.info())
print("Describing:",data.describe())

len_ham = len(data["label"][data.label=="ham"])
len_spam = len(data["label"][data.label=="spam"])

arr = np.array([len_ham,len_spam])
print("No of Ham:",len_ham,", No of spam:",len_spam)
plt.pie(arr,labels=["ham","spam"],explode=[0.2,0.0],shadow=True)
plt.show()

def text_preprocess(txt):
    txt=str(txt).lower()
    txt = txt.replace(",000,000","m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")

    words = word_tokenize(txt)
    processed_words=[]

    ps=PorterStemmer()
    for w in words:
        processed_words.append(ps.stem(w))
    txt = " ".join(processed_words)

    return txt


data["Preprocessed Text"] = data["messages"].apply(lambda text:text_preprocess(text))

print(data.head())

data["label"] = data.label.map({"ham":0,"spam":1})


X_train, X_test, y_train, y_test = train_test_split(data['Preprocessed Text'], 
                                                    data['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(data.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

count_vector = CountVectorizer()

training_data = count_vector.fit_transform(X_train)

testing_data = count_vector.transform(X_test)

mnb = MultinomialNB()

mnb.fit(training_data,y_train)

predictions = mnb.predict(testing_data)

print(predictions)

print('Accuracy score: ', format(accuracy_score(y_test, predictions))) 
print('Precision score: ', format(precision_score(y_test, predictions))) 
print('Recall score: ', format(recall_score(y_test, predictions))) 
print('F1 score: ', format(f1_score(y_test, predictions))) 
