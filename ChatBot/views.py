from django.shortcuts import render,redirect
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import os
from .chat_bot import *
from django.shortcuts import render
from django.http import HttpRequest
from django.db import connection
cur=connection.cursor()

from ChatBot.models import Message, Prompt

_INDEXPAGE = "index.html"

# Data reading
training = pd.read_csv(os.path.join(os.getcwd(),'models' , 'Training.csv'))
testing= pd.read_csv(os.path.join(os.getcwd(),'models' , 'Testing.csv'))

cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print (scores.mean())


model=SVC()
model.fit(x_train,y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def reset(req):
    os.system("python ChatBot/chat_bot.py")
    return redirect("index")

def contact(request):
    return render(request,'register.html')


def registration(request):
    if request.method=='POST':
        fname=request.POST['fname']
        lname=request.POST['lname']
        email=request.POST['email']
        pwd=request.POST['pwd']
        mno=request.POST['mno']
        addr=request.POST['addr']
        sql="select Email from chatbot_registration where  Email='%s'" %(email)
        # val=(email)
        cur.execute(sql)
        data=cur.fetchall()
        connection.commit()
        data=[j for i in data for j in i]
        if data ==[]:
            sql="insert into chatbot_registration(first_Name,Last_Name,Email,Password,Phone_Number,Address) values (%s,%s,%s,%s,%s,%s)"
            val=(fname,lname,email,pwd,mno,addr)
            cur.execute(sql,val)
            connection.commit()
            msg="Account created Successully"
            return render(request,'userLogin.html',{'msg':msg})
        msg="Details already Exists"
        return render(request, 'index.html',{'msg':msg})
    return render(request,'register.html')


def login(request):
    if request.method=='POST':
        name=request.POST['email']
        password=request.POST['pwd']
        sql="select * from chatbot_registration where Email='%s' and Password='%s'"%(name,password)
        cur.execute(sql)
        data=cur.fetchall()
        connection.commit()
        data=[i for i in data]
        if data !=[]:
            return render(request,'userDashboard.html')
        msg="Invalid credentials"
        return render(request, 'userLogin.html',{'msg':msg})
    return render(request,'userLogin.html')


def index(req:HttpRequest):
    if req.method == "POST":
        prompt = req.POST.get("input")
        try:
            reply = sec_predict(prompt)
        except:
            reply = sec_predict("")
        Message.objects.create(text=prompt)
        is_prompted = Prompt.objects.count() > 0
        if is_prompted:
            Message.objects.create(text=reply , is_bot=True)
        else:
            Message.objects.create(text="Hi There, What Are Your Symtoms", is_bot=True , is_asking=True)
            Prompt.objects.create()
        return redirect("index")
    messages = Message.objects.all()
    return render(req,_INDEXPAGE , {"messages" : messages})


def sec_predict(symptoms_exp):
    df = pd.read_csv(os.path.join(os.getcwd(),'models' , 'Training.csv'))
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1


    return rf_clf.predict([input_vector])[0]