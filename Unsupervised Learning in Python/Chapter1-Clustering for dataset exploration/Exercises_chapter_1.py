# ********** Clustering 2D points **********
# From the scatter plot of the previous exercise, you saw that the points seem to separate into 3 clusters. 
# You'll now create a KMeans model to find 3 clusters, and fit it to the data points from the previous exercise. 
# After the model has been fit, you'll obtain the cluster labels for some new points using the .predict() method.

# You are given the array points from the previous exercise, and also an array new_points.


# ********** Exercise Instrucions **********
# 1 - Import KMeans from sklearn.cluster.
# 2 - Using KMeans(), create a KMeans instance called model to find 3 clusters. To specify the number of clusters, use the n_clusters keyword argument.
# 3 - Use the .fit() method of model to fit the model to the array of points points.
# 4 - Use the .predict() method of model to predict the cluster labels of new_points, assigning the result to labels.
# 5 - Hit 'Submit Answer' to see the cluster labels of new_points.


# ********** Script **********

# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

# ------------------------------------------------------------------------------

# ********** Inspect your clustering **********
# Let's now inspect the clustering you performed in the previous exercise!

# A solution to the previous exercise has already run, so new_points is an array of points and labels is the array of their cluster labels.


# ********** Exercise Instrucions **********
# 1 - Import matplotlib.pyplot as plt.
# 2 - Assign column 0 of new_points to xs, and column 1 of new_points to ys.
# 3 - Make a scatter plot of xs and ys, specifying the c=labels keyword arguments to color the points by their cluster label. Also specify alpha=0.5.
# 4 - Compute the coordinates of the centroids using the .cluster_centers_ attribute of model.
# 5 - Assign column 0 of centroids to centroids_x, and column 1 of centroids to centroids_y.
# 6 - Make a scatter plot of centroids_x and centroids_y, using 'D' (a diamond) as a marker by specifying the marker parameter. Set the size of the markers to be 50 using s=50.


# ********** Script **********

# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to 
define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, 
centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker="D", s=50)
plt.show()

# ------------------------------------------------------------------------------

# ********** How many clusters of grain? **********
# In the video, you learned how to choose a good number of clusters for a dataset using the k-means inertia graph. 
# You are given an array samples containing the measurements (such as area, perimeter, length, and several others) of samples of grain. 
# What's a good number of clusters in this case?

# KMeans and PyPlot (plt) have already been imported for you.

# This dataset was sourced from the UCI Machine Learning Repository.


# ********** Exercise Instrucions **********
# 1 - For each of the given values of k, perform the following steps:
# Create a KMeans instance called model with k clusters.
# Fit the model to the grain data samples.
# Append the value of the inertia_ attribute of model to the list inertias.
#The code to plot ks vs inertias has been written for you, so hit 'Submit Answer' to see the plot!


# ********** Script **********

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# ------------------------------------------------------------------------------

# ********** Evaluating the grain clustering **********
# In the previous exercise, you observed from the inertia plot that 3 is a good number of clusters for the grain data. 
# In fact, the grain samples come from a mix of 3 different grain varieties: "Kama", "Rosa" and "Canadian". 
# In this exercise, cluster the grain samples into three clusters, and compare the clusters to the grain varieties using a cross-tabulation.

# You have the array samples of grain samples, and a list varieties giving the grain variety for each sample. 
# Pandas (pd) and KMeans have already been imported for you.


# ********** Exercise Instrucions **********
# 1 - Create a KMeans model called model with 3 clusters.
# 2 - Use the .fit_predict() method of model to fit it to samples and derive the cluster labels. Using .fit_predict() is the same as using .fit() followed by .predict().
# 3 - Create a DataFrame df with two columns named 'labels' and 'varieties', using labels and varieties, respectively, for the column values. This has been done for you.
# 4 - Use the pd.crosstab() function on df['labels'] and df['varieties'] to count the number of times each grain variety coincides with each cluster label. Assign the result to ct.
# 5 - Hit 'Submit Answer' to see the cross-tabulation!


# ********** Script **********

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

# ------------------------------------------------------------------------------

# ********** Scaling fish data for clustering **********
# You are given an array samples giving measurements of fish. Each row represents an individual fish. 
# The measurements, such as weight in grams, length in centimeters, and the percentage ratio of height to length, have very different scales.
# In order to cluster this data effectively, you'll need to standardize these features first. 
# In this exercise, you'll build a pipeline to standardize and cluster the data.

# These fish measurement data were sourced from the Journal of Statistics Education.


# ********** Exercise Instrucions **********
# 1 - Import:
# 1.1 - make_pipeline from sklearn.pipeline.
# 1.2 - StandardScaler from sklearn.preprocessing.
# 1.3 - KMeans from sklearn.cluster.
# 2 - Create an instance of StandardScaler called scaler.
# 3 - Create an instance of KMeans with 4 clusters called kmeans.
# 4 - Create a pipeline called pipeline that chains scaler and kmeans. To do this, you just need to pass them in as arguments to make_pipeline().


# ********** Script **********

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

# ------------------------------------------------------------------------------

# ********** Clustering the fish data **********
# You'll now use your standardization and clustering pipeline from the previous exercise to cluster the fish by their measurements, 
# and then create a cross-tabulation to compare the cluster labels with the fish species.

# As before, samples is the 2D array of fish measurements. 
# Your pipeline is available as pipeline, and the species of every fish sample is given by the list species.


# ********** Exercise Instrucions **********
# 1 - Import pandas as pd.
# 2 - Fit the pipeline to the fish measurements samples.
# 3 - Obtain the cluster labels for samples by using the .predict() method of pipeline.
# 4 - Using pd.DataFrame(), create a DataFrame df with two columns named 'labels' and 'species', using labels and species, respectively, for the column values.
# 5 - Using pd.crosstab(), create a cross-tabulation ct of df['labels'] and df['species'].


# ********** Script **********

# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'] )

# Display ct
print(ct)

# ------------------------------------------------------------------------------

# ********** Clustering stocks using KMeans **********
# In this exercise, you'll cluster companies using their daily stock price movements (i.e. the dollar difference between the closing and opening prices for each trading day). 
# You are given a NumPy array movements of daily price movements from 2010 to 2015 (obtained from Yahoo! Finance), where each row corresponds to a company, and each column corresponds to a trading day.

# Some stocks are more expensive than others. 
# To account for this, include a Normalizer at the beginning of your pipeline. T
# he Normalizer will separately transform each company's stock price to a relative scale before the clustering begins.

# Note that Normalizer() is different to StandardScaler(), which you used in the previous exercise. 
# While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing 
# the mean and scaling to unit variance, Normalizer() rescales each sample - here, each company's stock price - independently of the other.

# KMeans and make_pipeline have already been imported for you.


# ********** Exercise Instrucions **********
# 1 - Import Normalizer from sklearn.preprocessing.
# 2 - Create an instance of Normalizer called normalizer.
# 3 - Create an instance of KMeans called kmeans with 10 clusters.
# 4 - Using make_pipeline(), create a pipeline called pipeline that chains normalizer and kmeans.
# 5 - Fit the pipeline to the movements array.


# ********** Script **********

# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# ------------------------------------------------------------------------------

# ********** Which stocks move together? **********
# In the previous exercise, you clustered companies by their daily stock price movements. 
# So which company have stock prices that tend to change in the same way? 
# You'll now inspect the cluster labels from your clustering to find out.

# Your solution to the previous exercise has already been run. 
# Recall that you constructed a Pipeline pipeline containing a KMeans model and fit it to the NumPy array movements of daily stock movements. 
# In addition, a list companies of the company names is available.


# ********** Exercise Instrucions **********
# 1 - Import pandas as pd.
# 2 - Use the .predict() method of the pipeline to predict the labels for movements.
# 3 - Align the cluster labels with the list of company names companies by creating a DataFrame df with labels and companies as columns. This has been done for you.
# 4 - Use the .sort_values() method of df to sort the DataFrame by the 'labels' column, and print the result.
# 5 - Hit 'Submit Answer' and take a moment to see which companies are together in each cluster!


# ********** Script **********

# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))
