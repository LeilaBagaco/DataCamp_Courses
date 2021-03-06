# ********** Centering and scaling in a pipeline **********
# With regard to whether or not scaling is effective, the proof is in the pudding! 
# See for yourself whether or not scaling the features of the White Wine Quality dataset has any impact on its performance. 
# You will use a k-NN classifier as part of a pipeline that includes scaling, and for the purposes of comparison, a k-NN classifier trained on the unscaled data has been provided.

# The feature array and target variable array have been pre-loaded as X and y. 
# Additionally, KNeighborsClassifier and train_test_split have been imported from sklearn.neighbors and sklearn.model_selection, respectively.


# ********** Exercise Instructions **********
# 1 - Import the following modules:
# 1.1 - StandardScaler from sklearn.preprocessing.
# 1.2 - Pipeline from sklearn.pipeline.

# 2 - Complete the steps of the pipeline with StandardScaler() for 'scaler' and KNeighborsClassifier() for 'knn'.

# 3 - Create the pipeline using Pipeline() and steps.

# 4 - Create training and test sets, with 30% used for testing. Use a random state of 42.

# 5 - Fit the pipeline to the training set.

# 6 - Compute the accuracy scores of the scaled and unscaled models by using the .score() method inside the provided print() functions.


# ********** Script **********

# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))
