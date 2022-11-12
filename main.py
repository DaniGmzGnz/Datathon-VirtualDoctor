import contextlib
import csv
import re
import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree

warnings.filterwarnings("ignore", category=DeprecationWarning)

SEVERITY_THRESHOLD = 13


def _start():
    print("\n","-"*30, "Disease Prediction", "-"*30)
    print("\nWhat's your Name? ", end="-> ")
    name=input("")
    print("\nHello "+name+", you must answer some short questions in order to obtain a prediction on a disease you may have.")


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
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


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
    print("Are you experiencing any ...")
    secondary_symptoms=[]
    for symptom in list(symptoms_given):
        if symptom != inp_disease:
            print(" - "+symptom+"? ", end="")
        else:
            continue

        while True:
            inp = input("")
            if inp in ["yes", "no"]:
                break
            else:
                print("You must answer with yes/no: ", end="")

        if(inp == "yes"):
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

        print("\n\n", "-"*50, "PREDICTION INFORMATION", "-"*50)
        
        # If the two diseases found match show it
        if(present_disease[0] == second_prediction[0]):
            print("\nYou may have", present_disease[0])

            if present_disease[0] in info['description'].keys():
                print("\n"+info['description'][present_disease[0]])
            else:
                print("\nSorry, we don't have any useful description for this disease.")

            # readn(f"You may have {present_disease[0]}")
            # readn(f"{description_list[present_disease[0]]}")

        # If they don't match show both
        else:
            print("\nYou may have", present_disease[0], "or", second_prediction[0])
            if present_disease[0] in info['description'].keys():
                print("\n"+info['description'][present_disease[0]])
            else:
                print("Sorry, we don't have any useful description for this disease.")
            if second_prediction[0] in info['description'].keys():
                print("\n"+info['description'][second_prediction[0]])
            else:
                print("Sorry, we don't have any useful description for this disease.")

        # Show precautions for disease
        precaution_list = info['precaution'][present_disease[0]]
        print("\nYou should follow the next precautions: ")
        for j in precaution_list:
            print(" -",j)


def _get_init_symptom(feature_names):

    chk_dis = ",".join(feature_names).split(",")

    while True:
        # Get the initial symtpom.
        print("\nPlease, tell me the most important symptom you are experiencing: ", end="")
        inp_disease = input("")
        conf, cnf_dis = _get_related(chk_dis, inp_disease)

        # If symptom exists
        if conf==1:

            if len(cnf_dis) > 1:
                print("Here are some diseases related to your definition: ")

            for number, disease in enumerate(cnf_dis):
                if len(cnf_dis) > 1: print(" -",disease)

            # If there is more than one search related
            if number > 0:
                print("Please, write the disease you meant: ", end="")
                conf_inp = input("")
                inp_disease = conf_inp

            else:
                conf_inp = 0
                inp_disease = cnf_dis[conf_inp]

            break

        else:
            print("Enter valid symptom.")
    return inp_disease


def _get_num_days(info):
    while True:
        try:
            info['days'] = int(input("\nFrom how many days are you feeling this symptom? "))
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






