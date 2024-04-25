import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#imported XAI packages


def element_1():
    print("element 1")

def element_2():
    print("element 2")

def element_3():
    data_URL = "heartattackdata.csv"
    heart_data = pd.read_csv(data_URL)
    #imported data to be used

    heart_data.hist(bins=25,figsize=(10,10))
    #selects the amount of bins(bars) for each histogram
    """
    heart_data.hist(bins=25,figsize=(10,10),column=["Age","Gender"])
    #the column parameter can also be used to select specific columns for example above Age, and Gender
    """
    #plt.show()
    #create and display histograms from data

    X = heart_data.drop(columns="Result")
    #A new dataset X is created with the result column dropped as we want to use our data to predict this outcome
    y = heart_data["Result"]
    #This Result column is then used for our y dataset

    print(y.head)
    #Printing the head of the y array to troubleshoot whether the dataset has been correctly created

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,stratify =y,random_state = 13)
    #creates test train split, test_size being proportion of data to include in train split

    dt_clf = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 2)
    dt_clf.fit(X_train, y_train)
    #create decision tree fitting X and Y train data


    y_pred = dt_clf.predict(X_test)
    #makes predictions based on the test data

    print(classification_report(y_pred, y_test))
    #evaluates the predictions

    class_names = ["Heart Attack", "No Heart Attack"]
    #Set the class names

    feature_names = list(X_train.columns)
    #get the feature names

    fig = plt.figure(figsize=(25,20))
    _ = plot_tree(dt_clf, feature_names = feature_names, class_names = class_names, filled=True)
    plt.show()
    #plot decision tree and display

element_3()




"""
elementchoice = input ("Please select from the following application elements: \n 1: Element 1 \n 2: Element 2 \n 3: Element 3\n")
if elementchoice == "1":
    element_1()
elif elementchoice == "2":
    element_2()
elif elementchoice == "3":
    element_3()
else:
    elementchoice = input ("Please select 1, 2, or 3!\n")
"""

#simple if statements to select application element, subject to change as want to return back to menu after element has been explored