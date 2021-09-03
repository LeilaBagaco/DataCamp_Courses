# ********** Implementing logistic regression **********
# This is very similar to the earlier exercise where you implemented linear regression "from scratch" using scipy.optimize.minimize. 
# However, this time we'll minimize the logistic loss and compare with scikit-learn's LogisticRegression 
# (we've set C to a large value to disable regularization; more on this in Chapter 3!).

# The log_loss() function from the previous exercise is already defined in your environment, 
# and the sklearn breast cancer prediction dataset (first 10 features, standardized) is loaded into the variables X and y.

# ********** Exercise Instructions **********
# 1 - Input the number of training examples into range().

# 2 - Fill in the loss function for logistic regression.

# 3 - Compare the coefficients to sklearn's LogisticRegression.


# ********** Script **********

# The logistic loss, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        raw_model_output = w@X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X,y)
print(lr.coef_)
