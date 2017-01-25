# -*- coding:utf-8 -*-

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

# TODO: Compute desired values - replace each '?' with an appropriate expression/function call
n_students = student_data.shape[0]
n_features = student_data.shape[1] - 1
n_passed = student_data.groupby('passed').size()[1]
n_failed = student_data.groupby('passed').size()[0]
grad_rate = (n_passed * 100.0)/(n_passed + n_failed)
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# preparing data
# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows

# 清洗数据,如有yes/no就用1/0代替,如果缺失就删除或者用什么值代替
# Preprocess Feature Columns
# As you can see, there are several non-numeric columns that need to be converted! Many of them are simply yes/no, e.g.
# internet. These can be reasonably converted into 1/0 (binary) values.

# Other columns, like Mjob and Fjob, have more than two values, and are known as categorical variables.
# The recommended way to handle such a column is to create as many columns as possible values (e.g. Fjob_teacher,
# Fjob_other, Fjob_services, etc.), and assign a 1 to one of them and 0 to all others.
# These generated columns are sometimes called dummy variables, and we will use the pandas.get_dummies() function
# to perform this transformation.
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=num_train, random_state=100)
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])

# Train a model
import time

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

# TODO: Choose a model, import it and instantiate an object
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# Fit model to training data
train_classifier(clf, X_train, y_train)  # note: using entire training set here
print clf
# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score)
# Predict on test data
print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))
print len(X_train[:200])
# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

# TODO: Run the helper function above for desired subsets of training data
# Note: Keep the test set constant
train_predict(clf, X_train[:100], y_train[:100], X_test, y_test)
train_predict(clf, X_train[:200], y_train[:200], X_test, y_test)
train_predict(clf, X_train, y_train, X_test, y_test)


'''
Gaussian Naive Bayes (GaussianNB)
Decision Trees
Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
K-Nearest Neighbors (KNeighbors)
Stochastic Gradient Descent (SGDC)
Support Vector Machines (SVM)
Logistic Regression
'''
# TODO: Train and predict using two other models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

clf_rfc = RandomForestClassifier()
clf_svc = SVC()

# 预测的数据
train_predict(clf_rfc, X_train[:100], y_train[:100], X_test, y_test)
train_predict(clf_rfc, X_train[:200], y_train[:200], X_test, y_test)
train_predict(clf_rfc, X_train, y_train, X_test, y_test)

train_predict(clf_svc, X_train[:100], y_train[:100], X_test, y_test)
train_predict(clf_svc, X_train[:200], y_train[:200], X_test, y_test)
train_predict(clf_svc, X_train, y_train, X_test, y_test)

# choose best model
# TODO: Fine-tune your model and report the best F1 score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

# Setup parameters and cross validation for model optimization through Grid Search
parameters = {'gamma': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
              'C': [1, 10, 100, 200, 300, 400, 500, 600, 700]
             }


# 洗牌,混合
cv = StratifiedShuffleSplit(y_all, random_state=42)

gs = GridSearchCV(SVC(), parameters, cv=cv, scoring='f1_weighted')
gs.fit(X_all, y_all)

# Select the best settings for classifier
best_clf = gs.best_estimator_

# Fit the algorithm to the training data
print "Final Model: "
print best_clf
print '\n'
train_classifier(best_clf, X_train, y_train)

# Test algorithm's performance
print "F1 score for training set: {}".format(predict_labels(best_clf, X_train, y_train))
print "F1 score for test set: {}".format(predict_labels(best_clf, X_test, y_test))
