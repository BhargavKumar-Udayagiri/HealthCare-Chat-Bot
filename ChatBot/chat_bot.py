import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
from tkinter import *
warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('models/Training.csv')
testing= pd.read_csv('models/Testing.csv')
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
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
print (scores.mean())


model=SVC()
model.fit(x_train,y_train)
print("for svm: ")
print(model.score(x_test,y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        res = "You should take the consultation from doctor. "
        ChatLog.insert(END, res + '\n\n', 'Bot')

    else:
        res = "It might not be that bad but you should take precautions."
        ChatLog.insert(END, res + '\n\n', 'Bot')


def getDescription():
    global description_list
    with open('models/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getSeverityDict():
    global severityDictionary
    with open('models/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('models/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    # name=input("Name:")
    stR = "Please Enter your Name"

    return stR

def check_pattern(dis_list,inp):
    import re
    pred_list=[]
    ptr=0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:

        # print(f"comparing {inp} to {item}")
        if regexp.search(item):
            pred_list.append(item)
            # return 1,item
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return ptr,item

def sec_predict(symptoms_exp):
    df = pd.read_csv('models/Training.csv')
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


    return rf_clf.predict([input_vector])


def print_disease(node):
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero()
    # print(val)
    disease = le.inverse_transform(val[0])
    return disease


def get_node():
    global node, depth

    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]
        print(name, disease_input)
        if name == disease_input:
            val = 1
        else:
            val = 0
        if val <= threshold:
            print(node)
            node = tree_.children_left[node]
            depth = depth + 1
            get_node()
        else:
            symptoms_present.append(name)
            node = tree_.children_right[node]
            depth =depth + 1
            get_node()


def recurse():
    global flag_endloop, node, depth
    while True:
        num_days = int(ans)
        get_node()
        print("final_node", node)
        present_disease = print_disease(tree_.value[node])
        # print( "You may have " +  present_disease )
        red_cols = reduced_data.columns
        symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
        # dis_list=list(symptoms_present)
        # if len(dis_list)!=0:
        #     print("symptoms present  " + str(list(symptoms_present)))
        # print("symptoms given "  +  str(list(symptoms_given)) )

        res = "Are you experiencing any \n"
        ChatLog.insert(END, "Bot: " + res + '\n\n', 'Bot')
        symptoms_exp=[]
        for syms in list(symptoms_given):
            yield syms + "? \n"
            while True:
                inp=ans
                if(inp=="yes" or inp=="no"):
                    break
                else:
                    yield "provide proper answers i.e. (yes/no) : "
            if(inp=="yes"):
                symptoms_exp.append(syms)

        second_prediction=sec_predict(symptoms_exp)
        # print(second_prediction)
        calc_condition(symptoms_exp,num_days)
        if(present_disease[0]==second_prediction[0]):
            res = "You may have " + present_disease[0]
            ChatLog.insert(END, "Bot: " + res + '\n\n', 'Bot')

            res = description_list[present_disease[0]]
            ChatLog.insert(END, "Bot: " + res + '\n\n', 'Bot')

            # readn(f"You may have {present_disease[0]}")
            # readn(f"{description_list[present_disease[0]]}")

        else:
            res = "You may have " + present_disease[0] + " or " + second_prediction[0]
            ChatLog.insert(END, "Bot: " + str(res) + '\n\n', 'Bot')
            res = description_list[present_disease[0]]
            ChatLog.insert(END, "Bot: " + str(res) + '\n\n', 'Bot')
            res = description_list[second_prediction[0]]
            ChatLog.insert(END, "Bot: " + str(res) + '\n\n', 'Bot')

        # print(description_list[present_disease[0]])
        precution_list=precautionDictionary[present_disease[0]]
        res = "Take following measures : "
        ChatLog.insert(END, "Bot: " + res + '\n\n', 'Bot')
        for  i,j in enumerate(precution_list):
            res = str(i+1) + ")" + j
            ChatLog.insert(END, res + '\n\n', 'Bot')
        yield "Do you want to continue?"
        if ans == "yes":
            flag_endloop = False
            print("inside", flag_endloop)
            node, depth = 0, 1
            yield
            # res = tree_init_obj.__next__()
            # ChatLog.insert(END, res + '\n\n')
        else:
            quit()



def tree_to_code():
    global ans
    global feature_name, disease_input, num_days, flag_endloop, tree_, user_name

    yield "Hi, Please tell me your name."
    user_name = ans
    ChatLog.insert(END, "Bot: Hi," + user_name + '\n\n', 'Bot')

    while True:
        tree = clf
        feature_names = cols
        tree_ = tree.tree_

        # print(tree_)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]


        chk_dis=",".join(feature_names).split(",")



        # conf_inp=int()
        while True:

            yield "Enter the symptom you are experiencing"
            disease_input = ans
            conf,cnf_dis=check_pattern(chk_dis,disease_input)
            if conf==1:
                res = "searches related to input: "
                ChatLog.insert(END, "Bot: " + res + '\n\n', 'Bot')
                for num, it in enumerate(cnf_dis):
                    res =  str(num) + ")" + str(it)
                    ChatLog.insert(END, res + '\n\n')
                if num!=0:
                    yield "Select the one you meant (0 - {})".format(num)
                    conf_inp = ans
                else:
                    conf_inp=0
                disease_input=cnf_dis[int(conf_inp)]
                break
                # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
                # conf_inp = input("")
                # if(conf_inp=="yes"):
                #     break
            else:
                ChatLog.insert(END, "Enter valid symptom." + "\n", 'Bot')


        while True:
            try:
                flag_endloop = True
                yield "Okay. From how many days ? : "

                break
            except:
                yield "Enter number of days."






def send(event=None):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    global ans

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "\t\tYou: " + msg + '\n\n', 'You')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        ans = msg
        print("main", flag_endloop)
        if not flag_endloop:
            res = tree_init_obj.__next__()
            ChatLog.insert(END, "Bot: " + res + '\n\n', 'Bot')
        else:
            res = tree_obj.__next__()
            if flag_endloop:
                ChatLog.insert(END, "Bot: " + res + '\n\n', 'Bot')
            else:
                res = tree_init_obj.__next__()
                ChatLog.insert(END, "Bot: " + res + '\n\n', 'Bot')



ans = None
flag_endloop = None
tree_init_obj = tree_to_code()
node, depth = 0, 1
tree_obj = recurse()
symptoms_present = []
getSeverityDict()
getDescription()
getprecautionDict()
base = Tk()
base.title("Chat Bot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
ChatLog.tag_config('You', background="grey", foreground="black")
ChatLog.tag_config('Bot', background="black", foreground="white")
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
base.bind('<Return>', send)
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()


# getInfo()
# tree_to_code(clf,cols)

