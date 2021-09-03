# ********** KNN Classification **********
# In this exercise you'll explore a subset of the Large Movie Review Dataset.
# The variables X_train, X_test, y_train, and y_test are already loaded into the environment. 
# The X variables contain features based on the words in the movie reviews, and the y variables contain labels for whether the review sentiment is positive (+1) or negative (-1).

# This course touches on a lot of concepts you may have forgotten, so if you ever need a quick refresher, download the Scikit-Learn Cheat Sheet and keep it handy!

# ********** Exercise Instructions **********
# 1 - Create a KNN model with default hyperparameters.

# 2 - Fit the model.

# 3 - Print out the prediction for the test example 0.


# ********** Script **********

from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predict on the test features, print the results
pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)
