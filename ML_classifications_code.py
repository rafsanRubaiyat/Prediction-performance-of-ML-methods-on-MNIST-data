###############################################################################################
# STAT 603 Final Project 1 
# Prediction performance of different statistical modeling methods on analysis of the MNIST data
# Rafsan Siddiqui
###############################################################################################

# import required packages 
import pandas as pd
import numpy as np

# for logistic regression 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# for decision trees and random forest 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# for SVM 
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# for LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# for DNN 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# for CNN 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 

# import the required files 
train_binary_df = pd.read_csv("mnist_train_binary.csv")
train_counts_df = pd.read_csv("mnist_train_counts.csv")
test_binary_df = pd.read_csv("mnist_test_binary.csv")
test_counts_df = pd.read_csv("mnist_test_counts.csv")
test_counts_new_df = pd.read_csv("mnist_test_counts_new.csv")

# rename the columns for the compressed datasets 
new_column_names = ['label'] + ['x' + str(i) for i in range(1, 50)]

train_counts_df.columns = new_column_names
test_counts_df.columns = new_column_names
test_counts_new_df.columns = new_column_names

# Separate features and labels
X = train_counts_df.drop(columns=['label'])
y = train_counts_df['label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

########################################################################
# Multinomial Logistic Regression (with Lasso, Ridge, and Elastic Net) # 
########################################################################

# Use LogisticRegressionCV for cross-validation and hyperparameter tuning
# LogisticRegressionCV performs k-fold cross-validation to find the best C
model = LogisticRegressionCV(
    Cs=10,  # Number of values to try for C
    cv=5,  # Number of cross-validation folds
    penalty='l1',
    solver='saga',
    multi_class='multinomial',
    max_iter=3000,
    scoring='accuracy'
)

# Fit the model
model.fit(X_scaled, y)

# Best C value found
best_C = model.C_[0]
print(f'Best C value: {best_C}')

# Cross-validation score
cv_score = model.score(X_scaled, y)
print(f'Cross-validation accuracy: {cv_score}')

# Make predictions and evaluate on the same training set
y_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
print(f'Training set accuracy: {accuracy}')

##########################
# Ridge (L2) regularization
ridge_model = LogisticRegressionCV(
    Cs=10,  # Number of values to try for C
    cv=5,  # Number of cross-validation folds
    penalty='l2',  # L2 regularization
    solver='saga',  # Solver that supports L2 regularization
    multi_class='multinomial',  # Multinomial logistic regression
    max_iter=300,  # Maximum number of iterations
    scoring='accuracy'  # Scoring metric
)

ridge_model.fit(train_counts_df.drop(columns=['label']), train_counts_df['label'])

# Best C value found
best_C = ridge_model.C_[0]
print(f'Best C value: {best_C}')

# Cross-validation score
cv_score = ridge_model.score(X_scaled, y)
print(f'Cross-validation accuracy: {cv_score}')

# Make predictions and evaluate on the same training set
y_pred = ridge_model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
print(f'Training set accuracy: {accuracy}')

############################
# Elastic Net regularization
elastic_net_model = LogisticRegressionCV(
    Cs=10,  # Number of values to try for C
    cv=5,  # Number of cross-validation folds
    penalty='elasticnet',  # Elastic Net regularization
    solver='saga',  # Solver that supports Elastic Net regularization
    l1_ratios=[0.5],  # L1 ratio(s) to use in the Elastic Net regularization
    multi_class='multinomial',  # Multinomial logistic regression
    max_iter=300,  # Maximum number of iterations
    scoring='accuracy'  # Scoring metric
)

elastic_net_model.fit(train_counts_df.drop(columns=['label']), train_counts_df['label'])

# Best C value found
best_C = elastic_net_model.C_[0]
print(f'Best C value: {best_C}')

# Cross-validation score
cv_score = elastic_net_model.score(X_scaled, y)
print(f'Cross-validation accuracy: {cv_score}')

# Make predictions and evaluate on the same training set
y_pred = elastic_net_model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
print(f'Training set accuracy: {accuracy}')

####################################
# Decision Tree and Random Forest # 
###################################

# Decision Tree Classifier with GridSearchCV for hyperparameter tuning
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, n_jobs=-1, verbose=1)
dt_grid.fit(X, y)

# Best parameters and training the best model
dt_best_params = dt_grid.best_params_
dt_best_model = dt_grid.best_estimator_

# Cross-validation score for Decision Tree
dt_cv_scores = cross_val_score(dt_best_model, X, y, cv=5)
print("Decision Tree Classifier CV Scores:", dt_cv_scores)
print("Mean CV Score:", dt_cv_scores.mean())

# Random Forest Classifier with GridSearchCV for hyperparameter tuning
rf_params = {
    'n_estimators': [50, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, n_jobs=-1, verbose=1)
rf_grid.fit(X, y)

# Best parameters and training the best model
rf_best_params = rf_grid.best_params_
rf_best_model = rf_grid.best_estimator_

# Cross-validation score for Random Forest
rf_cv_scores = cross_val_score(rf_best_model, X, y, cv=5)
print("Random Forest Classifier CV Scores:", rf_cv_scores)
print("Mean CV Score:", rf_cv_scores.mean())

############################################
# Multi-class Support Vector Machine (SVM) # 
############################################

# SVM with GridSearchCV for hyperparameter tuning
svm_params = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [1, 0.1, 0.01, 0.001],
    'svc__kernel': ['linear', 'rbf', 'poly']
}

# Pipeline for scaling and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

svm_grid = GridSearchCV(pipeline, svm_params, cv=5, n_jobs=-1, verbose=1)
svm_grid.fit(X, y)

# Best parameters and training the best model
svm_best_params = svm_grid.best_params_
svm_best_model = svm_grid.best_estimator_

# Cross-validation score for SVM
svm_cv_scores = cross_val_score(svm_best_model, X, y, cv=5)
print("SVM Classifier CV Scores:", svm_cv_scores)
print("Mean CV Score:", svm_cv_scores.mean())

###############################
# SVM Misclassification Error #
###############################
# Separate features and labels
X_test = test_counts_df.drop(columns=['label'])
y_test = test_counts_df['label']

# Standardize the features using the same scaler fitted on the training data
X_test_scaled = svm_best_model.named_steps['scaler'].transform(X_test)

# Make predictions on the test data
y_pred = svm_best_model.named_steps['svc'].predict(X_test_scaled)

# Calculate the misclassification rate
accuracy = accuracy_score(y_test, y_pred)
misclassification_rate = 1 - accuracy

print(f"Test Accuracy: {accuracy}")
print(f"Test Misclassification Rate: {misclassification_rate}")

####################
# SVM Prediction 1 #
####################

# Standardize the features using the same scaler fitted on the training data
X_test_scaled = svm_best_model.named_steps['scaler'].transform(X_test)

# Make predictions on the test data
yhat1 = svm_best_model.named_steps['svc'].predict(X_test_scaled)

# Save predictions to file
pd.DataFrame(yhat1).to_csv("myfinal_prediction1.txt", header=False, index=False)


####################
# SVM Prediction 2 #
####################

# Separate features (labels are not used as they are all -1)
X_test_new = test_counts_new_df.drop(columns=['label'])

# Standardize the features using the same scaler fitted on the training data
X_test_new_scaled = svm_best_model.named_steps['scaler'].transform(X_test_new)

# Make predictions on the new test data
yhat2 = svm_best_model.named_steps['svc'].predict(X_test_new_scaled)

# Save new predictions to file
pd.DataFrame(yhat2).to_csv("myfinal_prediction2.txt", header=False, index=False)


##################################################
# Multi-class Linear Discriminant Analysis (LDA) #
##################################################

# Pipeline for scaling and LDA
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('lda', LinearDiscriminantAnalysis())  # Apply LDA
])

# Cross-validation score for LDA
lda_cv_scores = cross_val_score(pipeline, X, y, cv=5, n_jobs=-1)
print("LDA Classifier CV Scores:", lda_cv_scores)
print("Mean CV Score:", lda_cv_scores.mean())

#############################
# Deep Neural Network (DNN) #
#############################

# Define the neural network model
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap the Keras model so it can be used with scikit-learn
keras_model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# Standardize features and create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('keras', keras_model)  # Apply the Keras model
])

# Define cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=kfold, n_jobs=-1)
print("DNN Classifier CV Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

######################################
# Convolutional Neural Network (CNN) #
######################################

# prepare the training data
new_column_names = ['label'] + ['x' + str(i) for i in range(1, 785)]
train_binary_df.columns = new_column_names

# Separate features and labels
X = train_binary_df.drop(columns=['label'])
y = train_binary_df['label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data for CNN
X_reshaped = X.values.reshape(-1, 28, 28, 1)

# Define the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap the Keras model so it can be used with scikit-learn
keras_model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# Define cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(keras_model, X_reshaped, y, cv=kfold, n_jobs=-1)
print("CNN Classifier CV Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Train the model on the full training data
keras_model.fit(X_reshaped, y)

###############################
# CNN Misclassification Error #
###############################

# prepare the testing data 
test_binary_df.columns = new_column_names

# Separate features and labels
X_test = test_binary_df.drop(columns=['label'])
y_test = test_binary_df['label']

# Standardize the features using the same scaler fitted on the training data
X_test_scaled = scaler.transform(X_test)

# Reshape the data for CNN
X_test_reshaped = X_test.values.reshape(-1, 28, 28, 1)

# Make predictions on the test data
y_pred = keras_model.predict(X_test_reshaped)

# Calculate the misclassification rate
accuracy = accuracy_score(y_test, y_pred)
misclassification_rate = 1 - accuracy

print(f"Test Accuracy: {accuracy}")
print(f"Test Misclassification Rate: {misclassification_rate}")