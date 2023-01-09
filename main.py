# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Import the data
raw_data_df = pd.read_csv('data.csv')

# Investigate observations
raw_data_df.info()

# Drop unnecessary columns
raw_data_df = raw_data_df.drop(['id','Unnamed: 32'], axis=1)

# Separate data into matrix of features and dependent variable vector
X = raw_data_df.iloc[:, 1:].values
y = raw_data_df.iloc[:, 0].values

le = LabelEncoder()
y = le.fit_transform(y)

# Get train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create model object
models = [GaussianNB, LogisticRegression, KNeighborsClassifier, SVC,
          DecisionTreeClassifier, RandomForestClassifier, XGBClassifier]

model_names = ['NaiveBayes', 'LogisticRegression', 'KNN', 'SVC',
               'DecisionTree', 'RandomForest', 'XGBoost']

for model, model_name in zip(models, model_names):
    
    if model_name == 'XGBoost':
        classifier = model(use_label_encoder=False, eval_metric='logloss')
        classifier.fit(X_train, y_train)
    else:
        classifier = model()
        classifier.fit(X_train, y_train)

    # Train the model
    classifier.fit(X_train, y_train)

    # Make the confusion matrix
    # y_pred = classifier.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print("\nConfusion Matrix" + " (" + model_name + ")")
    # print("----------------")
    # print(cm)
    # print("\nAccuracy score: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

    # Apply K-fold cross validation
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("\nCross Validation Method" + " (" + model_name + ")")
    print("-----------------------")
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))