# ********** Bringing it all together I: Pipeline for classification **********
# It is time now to piece together everything you have learned so far into a pipeline for classification!
# Your job in this exercise is to build a pipeline that includes scaling and hyperparameter tuning to classify wine quality.

# You'll return to using the SVM classifier you were briefly introduced to earlier in this chapter.
# The hyperparameters you will tune are C and gamma. C controls the regularization strength. 
# It is analogous to the C you tuned for logistic regression in Chapter 3, while gamma controls the kernel coefficient:
# Do not worry about this now as it is beyond the scope of this course.

# The following modules and functions have been pre-loaded: 
# Pipeline, SVC, train_test_split, GridSearchCV, classification_report, accuracy_score.
# The feature and target variable arrays X and y have also been pre-loaded.


# ********** Exercise Instructions **********
# 1 - Setup the pipeline with the following steps:
# 1.1 - Scaling, called 'scaler' with StandardScaler().
# 1.2 - Classification, called 'SVM' with SVC().

# 3 - Specify the hyperparameter space using the following notation: 'step_name__parameter_name'. Here, the step_name is SVM, and the parameter_names are C and gamma.

# 4 - Create training and test sets, with 20% of the data used for the test set. Use a random state of 21.

# 5 - Instantiate GridSearchCV with the pipeline and hyperparameter space and fit it to the training set. Use 3-fold cross-validation (This is the default, so you don't have to specify it).

# 6 - Predict the labels of the test set and compute the metrics. The metrics have been computed for you.


# ********** Script **********

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
