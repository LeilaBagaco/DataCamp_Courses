# ********** Sentiment analysis for movie reviews **********
# In this exercise you'll explore the probabilities outputted by logistic regression on a subset of the Large Movie Review Dataset.

# The variables x and y are already loaded into the environment.
# x contains features based on the number of times words appear in the movies review,
# and y contains labels for wether the review sentiment is positive (+1) or negative (-1).

# ********** Exercise Instructions **********
# 1 - Train a logistic regression model on the movie review data.

# 2 - Predict the probabilities of negative vs. positive for the two given reviews.

# 3 - Feel free to write your own reviews and get probabilities for those too!


# ********** Script **********

# Instantiate logistic regression and train
lr = LogisticRegression()
lr.fit(X, y)

# Predict sentiment for a glowing review
review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])

# Predict sentiment for a poor review
review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])
