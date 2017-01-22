# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
import pylab as pl
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV




# 1.load data
def load_data():
    boston = pd.read_csv('housing.csv')
    return boston

'''
    也可以从sklearn中下载数据,但是有数据些许的不同,如要使用多加注意
    boston = datasets.load_boston()
    housing_prices = boston.target
    housing_features = boston.data

'''

# 2.统计housing_data
def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    housing_prices = city_data['MEDV']
    housing_features = city_data.drop('MEDV',axis=1)

    # Size of data
    print "No. of data points: {}".format(housing_features.shape[0])

    # Number of features
    print "No. of features: {}".format(housing_features.shape[1])

    # Minimum housing price
    print "Minimum housing price: {}".format(min(housing_prices))

    # Maximum housing price
    print "Maximum housing price: {}".format(max(housing_prices))

    # Mean housing price
    print "Mean housing price: {}".format(np.mean(housing_prices))

    # Calculate median?
    print "Median housing price: {}".format(np.median(housing_prices))

    # Calculate standard deviation?
    print "Standard deviation of housing prices: {}".format(np.std(housing_prices))

# 验证预测的错误率
def performance_metric(label, prediction):
    """Calculate and return the appropriate performance metric."""

    mse = mean_squared_error(label, prediction)

    return mse

# 切分数据分别用与训练和验证
def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into training and testing set."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.drop('MEDV', axis=1), city_data['MEDV']

    # Split up the dataset between training and testing
    # 分离训练数据和验证数据
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, y_train, X_test, y_test


# 学习曲线
def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.linspace(1, len(X_train), 50)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        s = int(s)
        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)

# 绘制学习曲线
def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


# 训练模型并预测房价
def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.drop('MEDV', axis=1), city_data['MEDV']

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    # Setup parameters and scores for model optimization through Grid Search
    parameters = {'max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    gs = GridSearchCV(regressor, parameters, scoring=scorer)
    gs.fit(X, y)

    # Select the best settings for regressor
    reg = gs.best_estimator_

    # Fit the learner to the training data
    print "Final Model: "
    print reg.fit(X, y)

    # Use the model to predict the output of a particular sample
    x = [[11.95, 0.00, 18.100], [0, 0.6590, 5.6090], [90.00, 1.385, 24], [680.0, 20.20, 332.09]]
    y = reg.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()
    # print city_data

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # # Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()
