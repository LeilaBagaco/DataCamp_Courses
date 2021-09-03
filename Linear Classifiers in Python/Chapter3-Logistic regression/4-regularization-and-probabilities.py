# ********** Regularization and probabilities **********
# In this exercise, you will observe the effects of changing the regularization strength on the predicted probabilities.

# A 2D binary classification dataset is already loaded into the environment as X and y.


# ********** Exercise Instructions **********
# 1 - Compute the maximum predicted probability.
# 1.1 - Run the provided code and take a look at the plot.

# 2 - Create a model with C=0.1 and examine how the plot and probabilities change.


# ********** Script **********

# Set the regularization strength / change c to 0.1 in order to complete exercise 2
model = LogisticRegression(C=1)

# Fit and plot
model.fit(X,y)
plot_classifier(X,y,model,proba=True)

# Predict probabilities on training points
prob = model.predict_proba(X)
print("Maximum predicted probability", np.max(prob))
