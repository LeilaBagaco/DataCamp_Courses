# ********** Changing the model coefficients **********
# When you call fit with scikit-learn, the logistic regression coefficients are automatically learned from your dataset. 
# In this exercise you will explore how the decision boundary is represented by the coefficients. 
# To do so, you will change the coefficients manually (instead of with fit), and visualize the resulting classifiers.

# A 2D dataset is already loaded into the environment as X and y, along with a linear classifier object model.

# ********** Exercise Instructions **********
# 1 - Set the two coefficients and the intercept to various values and observe the resulting decision boundaries.

# 2 - Try to build up a sense of how the coefficients relate to the decision boundary.

# 3 - Set the coefficients and intercept such that the model makes no errors on the given training data.


# ********** Script **********

# Set the coefficients
model.coef_ = np.array([[-1,1]])
model.intercept_ = np.array([-3])

# Plot the data and decision boundary
plot_classifier(X,y,model)

# Print the number of errors
num_err = np.sum(y != model.predict(X))
print("Number of errors:", num_err)
