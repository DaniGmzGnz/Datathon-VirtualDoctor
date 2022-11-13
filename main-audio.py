import pyttsx3
import speech_recognition as sr
import webbrowser 
import datetime 

import contextlib
import csv
import re
import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree

from colored import fg

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEVERITY_THRESHOLD = 13
CYAN = '\033[94m'
CEND = '\033[0m'
CRED = '\033[91m'
CGREEN = '\033[92m'
 
# this method is for taking the commands
# and recognizing the command from the
# speech_Recognition module we will use
# the recongizer method for recognizing
def takeCommand():
 
    r = sr.Recognizer()
 
    # from the speech_Recognition module
    # we will use the Microphone module
    # for listening the command
    with sr.Microphone() as source:
        print(CYAN+'\nListening...'+CEND)
         
        # seconds of non-speaking audio before
        # a phrase is considered complete
        r.pause_threshold = 0.7
        audio = r.listen(source)
        # Now we will be using the try and catch
        # method so that if sound is recognized
        # it is good else we will have exception
        # handling
        try:
            print(CYAN+'Recognizing...'+CEND)
             
            # for Listening the command in indian
            # english we can also use 'hi-In'
            # for hindi recognizing
            Query = r.recognize_google(audio, language='en-en')
            print(CGREEN+"User: "+CEND +Query)
             
        except Exception as e:
            print(e)
            print(CRED+"I did not understand you. Please, repeat your answer."+CEND)
            return "None"
         
        return Query
 
def speak(audio):
     
    engine = pyttsx3.init()
    # getter method(gets the current value
    # of engine property)
    #voices = engine.getProperty('voices')
     
    # setter method .[0]=male voice and
    # [1]=female voice in set Property.
    #engine.setProperty('voice', voices[0].id)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-30)
     
    # Method for the speaking of the assistant
    engine.say(audio) 
     
    # Blocks while processing all the currently
    # queued commands
    engine.runAndWait()
 
def tellDay():
     
    # This function is for telling the
    # day of the week
    day = datetime.datetime.now().weekday() + 1

    #this line tells us about the number
    # that will help us in telling the day
    Day_dict = {1: 'Monday', 2: 'Tuesday',
                3: 'Wednesday', 4: 'Thursday',
                5: 'Friday', 6: 'Saturday',
                7: 'Sunday'}

    if day in Day_dict:
        day_of_the_week = Day_dict[day]
        print(day_of_the_week)
        speak("The day is " + day_of_the_week)
 
 
def tellTime():
     
    # This method will give the time
    time = str(datetime.datetime.now())
     
    # the time will be displayed like
    # this "2020-06-05 17:50:14.582630"
    #nd then after slicing we can get time
    print(time)
    hour = time[11:13]
    min = time[14:16]
    speak("The time is sir" + hour + "Hours and" + min + "Minutes")   
 
def Hello():
     
    # This function is for when the assistant
    # is called it will say hello and then
    # take query
    speak("hello sir I am your desktop assistant. Tell me how may I help you")


def _start():
    print("\n","-"*30, "Disease Prediction", "-"*30)
    print(CYAN+"VirtualDoctor: "+CEND+"What's your Name? ")
    speak("What's your Name?")
    # name=input("")
    name = takeCommand().lower()
    print(CYAN+"VirtualDoctor: "+CEND+"Hello "+name+", you must answer some short questions in order to obtain a prediction on a disease you may have")
    speak("Hello "+name+", you must answer some short questions in order to obtain a prediction on a disease you may have.")
    # print("\nHello "+name+", you must answer some short questions in order to obtain a prediction on a disease you may have.")


def _load_data() -> tuple[pd.DataFrame, pd.Series]:

    training = pd.read_csv('data/training_data.csv')
    
    cols = training.columns
    feature_names = cols[:-1]

    data = training[feature_names]
    target = training['prognosis']
    

    disease_names = training.groupby(training['prognosis']).max() # Disease Names

    return training, data, target, feature_names, disease_names


def _train_models(dataset, data, target) -> DecisionTreeClassifier:

    # Label Enconding
    le = preprocessing.LabelEncoder()
    le.fit(target)
    target = le.transform(target)

    # Search Tree model
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)
    clf_tree  = DecisionTreeClassifier()
    clf_tree = clf_tree.fit(x_train, y_train)

    # Model metrics
    scores = cross_val_score(clf_tree, x_test, y_test, cv=4)

    # Seconday Prediction model
    data = dataset.iloc[:, :-1]
    target = dataset['prognosis']
    x_train, _, y_train, _ = train_test_split(data, target, test_size=0.3, random_state=20)
    clf_pred = DecisionTreeClassifier()
    clf_pred = clf_pred.fit(x_train, y_train)


    return {'model_tree': clf_tree, 'model_pred': clf_pred}, le, scores


def _load_information() -> dict:
    description = {}
    severity = {}
    precaution = {}

    # Get description info
    with open('data/description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description={row[0]:row[1]}
            description |= _description

    # Get severity info
    with open('data/severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with contextlib.suppress(Exception):
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severity |= _diction

    # Get precaution info
    with open('data/precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precaution |= _prec

    return {
        'description': description,
        'severity': severity,
        'precaution': precaution
    }

def _get_severity(exp, info):   # sourcery skip: avoid-builtin-shadow
    sum=0
    for item in exp:
         sum = sum + info['severity'][item]

    if((sum * info['days']) / (len(exp) + 1) > SEVERITY_THRESHOLD):
        print(CYAN+"VirtualDoctor: "+CEND+"You should take the consultation from doctor. ")
        speak("You should take the consultation from doctor. ")
    else:
        print(CYAN+"VirtualDoctor: "+CEND+"It might not be that bad but you should take precautions.")
        speak("It might not be that bad but you should take precautions.")


def _get_related(dis_list, inp):

    inp = inp.replace(' ','_')
    regexp = re.compile(str(inp))

    found = [item for item in dis_list if regexp.search(item)]
    
    return (1, found) if found else (0, [])


def _secondary_prediction(model, symptoms_exp):
    df = pd.read_csv('data/training_data.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)


    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return model.predict([input_vector])


def _get_disease(value, le):
    disease = le.inverse_transform(value[0].nonzero()[0])
    return list(map(lambda x: x.strip(), list(disease)))


def _get_secondary_symptoms(inp_disease, symptoms_given):
    
    secondary_symptoms=[]
    for symptom in list(symptoms_given):
        if symptom == inp_disease:
            continue

        print(CYAN+"VirtualDoctor: "+CEND+" Are you experiencing any "+symptom+"? ")
        speak("Are you experiencing any" +symptom.replace("_", " "))
        while True:
            # inp = input("")
            while True:
                inp = takeCommand()
                if inp in ['yes','yes yes','yeah','ye','no','nope','no no']: break
                print(CYAN+"VirtualDoctor: "+CEND+"You must answer with yes or no: ")
                speak("You must answer with yes or no: ")
            break

        if(inp in ['yes','yes yes', 'yeah', 'ye']):
            secondary_symptoms.append(symptom)
    return secondary_symptoms


def _recursion(tree_, info, inp_disease, disease_names, node, depth) -> None:

    # If the current feature is undefined.
    if tree_.feature[node] != _tree.TREE_UNDEFINED:

        name = info['feature_name'][node]
        threshold = tree_.threshold[node]

        val = 1 if name == inp_disease else 0

        if val <= threshold:
            _recursion(tree_, info, inp_disease, disease_names, tree_.children_left[node], depth + 1)
        else:
            _recursion(tree_, info, inp_disease, disease_names, tree_.children_right[node], depth + 1)

    else:
        # Get name of current disease
        present_disease = _get_disease(tree_.value[node], info['le'])
        red_cols = disease_names.columns

        # Get symptoms related to primary disease
        symptoms_given = red_cols[disease_names.loc[present_disease].values[0].nonzero()]

        # Get secondary symptoms of the user
        secondary_symptoms = _get_secondary_symptoms(inp_disease, symptoms_given)
        # Calculate and show gravity of symptoms
        _get_severity(secondary_symptoms, info)

        # Predict a second disease.
        second_prediction = _secondary_prediction(info['model'], secondary_symptoms)

        #print("\n\n", "-"*50, "PREDICTION INFORMATION", "-"*50)
        
        # If the two diseases found match show it
        if(present_disease[0] == second_prediction[0]):
            print(CYAN+"VirtualDoctor: "+CEND+"You may have " + present_disease[0])
            speak("You may have"+present_disease[0])

            if present_disease[0] in info['description'].keys():
                print(CYAN+"VirtualDoctor: "+CEND+info['description'][present_disease[0]])
            else:
                print(CYAN+"VirtualDoctor: "+CEND+"Sorry, we don't have any useful description for this disease.")

            # readn(f"You may have {present_disease[0]}")
            # readn(f"{description_list[present_disease[0]]}")

        # If they don't match show both
        else:
            print(CYAN+"VirtualDoctor: "+CEND + "You may have "+ present_disease[0] +"or"+second_prediction[0])
            speak("You may have"+present_disease[0]+" or "+second_prediction[0])

            if present_disease[0] in info['description'].keys():
                print(CYAN+"VirtualDoctor: "+CEND +info['description'][present_disease[0]])
            else:
                print(CYAN+"VirtualDoctor: "+CEND+"Sorry, we don't have any useful description for this disease.")
            if second_prediction[0] in info['description'].keys():
                print(CYAN+"VirtualDoctor: "+CEND+info['description'][second_prediction[0]])
            else:
                print(CYAN+"VirtualDoctor: "+CEND+"Sorry, we don't have any useful description for this disease.")

        # Show precautions for disease
        precaution_list = info['precaution'][present_disease[0]]
        print(CYAN+"VirtualDoctor: "+CEND+"You should follow the next precautions: ")
        speak("You should follow the next precautions")
        for j in precaution_list:
            print(" -", j)
            speak(j)


def _get_init_symptom(feature_names):

    chk_dis = ",".join(feature_names).split(",")

    while True:
        # Get the initial symtpom.
        print(CYAN+"VirtualDoctor: "+CEND+"Please, tell me the most important symptom you are experiencing: ")
        speak("Please, tell me the most important symptom you are experiencing: ")
        # inp_disease = input("")
        inp_disease = takeCommand()
        conf, cnf_dis = _get_related(chk_dis, inp_disease)

        # If symptom exists
        if conf==1:

            if len(cnf_dis) > 1:
                print(CYAN+"VirtualDoctor: "+CEND+"Here are some diseases related to your definition: ")

            for number, disease in enumerate(cnf_dis):
                if len(cnf_dis) > 1: print(" -", disease)

            # If there is more than one search related
            if number > 0:
                print(CYAN+"VirtualDoctor: "+CEND+"Please, write the disease you meant: ")
                speak("Please, say the disease you meant: ")
                conf_inp = takeCommand()
                inp_disease = conf_inp

            else:
                conf_inp = 0
                inp_disease = cnf_dis[conf_inp]

            break

        else:
            print(CYAN+"VirtualDoctor: "+CEND + "Enter valid symptom.")
    return inp_disease


def _get_num_days(info):
    while True:
        try:
            print(CYAN+"VirtualDoctor: "+CEND+"From how many days are you feeling this symptom? ")
            speak("From how many days are you feeling this symptom? ")
            text = takeCommand()
            
            info['days'] = int(text)
            
            return info

        except Exception:
            print("Enter valid input.")


def _predict(models, info, feature_names, disease_names) -> None:
    
    info['model'] = models['model_pred']
    
    # SKLearn trained Tree
    tree = models['model_tree'].tree_

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]


    inp_disease = _get_init_symptom(feature_names)

    # Get number of days of the important disease
    info = _get_num_days(info)

    # Recursion to ask for another secondary diseases
    info['feature_name'] = feature_name
    _recursion(tree, info, inp_disease, disease_names, 0, 1)


def main() -> None:

    _start()

    dataset, data, target, feature_names, disease_names = _load_data()

    models, le, scores = _train_models(dataset, data, target)

    info = _load_information()

    info['le'] = le

    _predict(models, info, feature_names, disease_names)


main()