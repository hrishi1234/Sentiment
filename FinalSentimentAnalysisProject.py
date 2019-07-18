# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:34:56 2019

@author: Devanshi Gupta
"""


import pandas as pd
import numpy as np

dataset = pd.read_csv("C:/Users/HRISHI/Downloads/Sentiment.csv")
dataset1 = dataset[['text','sentiment']]
dataset1

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
data1 = []
for i in range(10725):
    review =  dataset['text'][i]
    review = re.sub('[^a-zA-Z]', ' ',review)   
    review = review.lower()
    review = review.split()  
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data1.append(review)
  
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)   
x = cv.fit_transform(data1).toarray()
y = dataset.iloc[:,5].values
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y = lb.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
 
model.add(Dense(output_dim=800,init="uniform",activation='relu',input_dim=3000))
model.add(Dense(output_dim=600,init="uniform",activation='relu'))
model.add(Dense(output_dim=1,init="uniform",activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=30,batch_size=32)

y_pred =model.predict(x_test)
r = "Me reading my family's comments about how great the #GOPDebate was http://t.co/gIaGjPygXZ"
review1 = re.sub('[^a-zA-Z]', ' ',r)   
review1 = review1.lower()
review1 = review1.split()  
review1 = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
review1 = ' '.join(review1)
ypred2 = model.predict(cv.transform([review1]))


from tkinter import *

top = Tk()
top.geometry("550x300+300+150")
top.resizable(width=True, height=True)

L1 = Label(top, text="write your comment")
L1.pack()
E1 = Entry(top, bd =5)
E1.pack()

def predict():
    print("Prediction on progress..")
    entered_input=E1.get()
    print("Entered Input",entered_input)
    print(type(entered_input))
    
    prediction = model.predict(cv.transform([entered_input]))
    print(prediction)
    if(prediction>0.5):
        p1="Positive Comment"
    else:
        p1="Negative Comment"
    
    L2 = Label(top, text="Entered Text: "+ entered_input)
    L2.pack()
    L2 = Label(top, text="Prediction: "+ p1 )
    L2.pack()
B = Button(top, text ="Predict", command =predict)
B.pack()
top.mainloop()

















