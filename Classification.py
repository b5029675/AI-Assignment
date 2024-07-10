import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#KNN CLASSIFICATION

#store file location in variable
file = "mushroom_poisonous.csv"

#read the CSV file
dataset = pd.read_csv(file, sep=',')

#print the head of the dataset, ensuring it has been loading correctly
print (dataset.head())

#The same data is going to be used as I'll be exploring the differences between the two classifiers

#assigns the columns of the CSV file, excluding the column we are attempting to predict
feature_columns = ["cap-diameter","cap-shape","gill-attachment","gill-color","stem-height","stem-width","stem-color","season"]

#Create our first dataset from these columns
X = dataset[feature_columns].values

#assign the column we want to predict to the y dataset
y= dataset["class"]

#create test and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#scales the dataset to make values more easily comparable
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#Creating graph for optimal K
# creating list of K for KNN
k_list = list(range(1,50,2))
# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]


plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)

# finding best k
best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)


#Create KNN classifier
classifier = KNeighborsClassifier(n_neighbors = 3)
#Fit the data
classifier.fit(X_train, y_train)

#predict the value of y, the class
y_predict = classifier.predict(X_test)

#print out the confusion matrix and the classification report
print("Confusion Matrix:", confusion_matrix(y_test, y_predict))
print("Classification Report:" , classification_report(y_test, y_predict))

#Confusion matrix Display
predictions = classifier.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_,)
disp.plot(cmap="Reds", values_format='')
disp.ax_.set_title("Gradient Boost Confusion Matrix")

#Displaying accuracy
accuracy = accuracy_score(y_test, y_predict)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')


#NAIVE BAYES

#store file location in variable
file = "mushroom_poisonous.csv"

#read the CSV file
dataset = pd.read_csv(file, sep=',')

#The same data is going to be used as I'll be exploring the differences between the two classifiers

feature_columns = ["cap-diameter", "cap-shape", "gill-attachment", "gill-color", "stem-height", "stem-width", "stem-color", "season"]

#Create X2 dataset for NB classifier
X2 = dataset[feature_columns].values

#assign the class column
y2= dataset["class"].values

#create test and train subsets for NB classifier
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3)


scaler = StandardScaler()
scaler.fit(X2_train)
X2_train = scaler.transform(X2_train)
X2_test = scaler.transform(X2_test)

#initialise NB classifier
NBclassifier = GaussianNB()

NBclassifier.fit(X2_train, y2_train)

#make predictions with classifier
y2_predict = NBclassifier.predict(X2_test)

#Evaluate accuracy
print("Confusion Matrix:", confusion_matrix(y2_test, y2_predict))
print("Classification Report:", classification_report(y2_test, y2_predict))

#Confusion matrix Display Naive Bayes
predictions = NBclassifier.predict(X2_test)
cm = confusion_matrix(y2_test, predictions, labels=NBclassifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=NBclassifier.classes_,)
disp.plot(cmap="Blues", values_format='')
disp.ax_.set_title("Gradient Boost Confusion Matrix")

accuracy = accuracy_score(y2_test, y2_predict)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

# Calculate the correlation matrix to view whether data is appropriate for naive bayes to determine why the results are so different
# as in Naive Bayes we assume that all features are independent given the class
correlation_matrix = dataset.corr()

print(correlation_matrix)


#dropping the stem-width only slightly improved the accuracy of naive bayes

#investigating why there is such a discrepancy between the 0 and 1 accuracy

# Print class distribution in the training set
print("Training Set Class Distribution:")
print(y_train.value_counts())

# Print class distribution in the testing set
print("\nTesting Set Class Distribution:")
print(y_test.value_counts())

plt.show()