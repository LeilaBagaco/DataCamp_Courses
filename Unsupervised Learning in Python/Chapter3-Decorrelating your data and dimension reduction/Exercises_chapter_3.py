# ********** Correlated data in nature *********
#You are given an array grains giving the width and length of samples of grain. You suspect that width and length will be correlated. 
# To confirm this, make a scatter plot of width vs length and measure their Pearson correlation.


# ********** Exercise Instructions **********
# 1 - Import:
# 1.1 - matplotlib.pyplot as plt.
# 1.2 - pearsonr from scipy.stats.
# 2 - Assign column 0 of grains to width and column 1 of grains to length.
# 3 - Make a scatter plot with width on the x-axis and length on the y-axis.
# 4 - Use the pearsonr() function to calculate the Pearson correlation of width and length.


# ********** Script **********

# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)


# -------------------------------------------------------------------------------

# ********** Decorrelating the grain measurements with PCA **********
# You observed in the previous exercise that the width and length measurements of the grain are correlated. 
# Now, you'll use PCA to decorrelate these measurements, then plot the decorrelated points and measure their Pearson correlation.


# ********** Exercise Instructions **********
# 1 - Import PCA from sklearn.decomposition.
# 2 - Create an instance of PCA called model.
# 3 - Use the .fit_transform() method of model to apply the PCA transformation to grains. Assign the result to pca_features.
# 4 - The subsequent code to extract, plot, and compute the Pearson correlation of the first two columns pca_features has been written for you, so hit 'Submit Answer' to see the result!


# ********** Script **********

# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)


# ------------------------------------------------------------------------------

# ********** The first principal component **********
# The first principal component of the data is the direction in which the data varies the most. 
# In this exercise, your job is to use PCA to find the first principal component of the length and width measurements of the grain samples, and represent it as an arrow on the scatter plot.

# The array grains gives the length and width of the grain samples. PyPlot (plt) and PCA have already been imported for you.


# ********** Exercise Instructions **********
# 1 - Make a scatter plot of the grain measurements. This has been done for you.
# 2 - Create a PCA instance called model.
# 3 - Fit the model to the grains data.
# 4 - Extract the coordinates of the mean of the data using the .mean_ attribute of model.
# 5 - Get the first principal component of model using the .components_[0,:] attribute.
# 6 - Plot the first principal component as an arrow on the scatter plot, using the plt.arrow() function. You have to specify the first two arguments - mean[0] and mean[1].


# ********** Script **********

# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()


# ------------------------------------------------------------------------------

# ********** Variance of the PCA features **********
# The fish dataset is 6-dimensional. But what is its intrinsic dimension? Make a plot of the variances of the PCA features to find out. 
# As before, samples is a 2D array, where each row represents a fish. You'll need to standardize the features first.


# ********** Exercise Instructions **********
# 1 - Create an instance of StandardScaler called scaler.
# 2 - Create a PCA instance called pca.
# 3 - Use the make_pipeline() function to create a pipeline chaining scaler and pca.
# 4 - Use the .fit() method of pipeline to fit it to the fish samples samples.
# 5 - Extract the number of components used using the .n_components_ attribute of pca. Place this inside a range() function and store the result as features.
# 6 - Use the plt.bar() function to plot the explained variances, with features on the x-axis and pca.explained_variance_ on the y-axis.


# ********** Script **********

# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


# ------------------------------------------------------------------------------

# ********** Dimension reduction of the fish measurements **********
# In a previous exercise, you saw that 2 was a reasonable choice for the "intrinsic dimension" of the fish measurements.
# Now use PCA for dimensionality reduction of the fish measurements, retaining only the 2 most important components.

# The fish measurements have already been scaled for you, and are available as scaled_samples.


# ********** Exercise Instructions **********
# 1 - Import PCA from sklearn.decomposition.
# 2 - Create a PCA instance called pca with n_components=2.
# 3 - Use the .fit() method of pca to fit it to the scaled fish measurements scaled_samples.
# 4 - Use the .transform() method of pca to transform the scaled_samples. Assign the result to pca_features.


# ********** Script **********

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)


# ------------------------------------------------------------------------------

# ********** A tf-idf word-frequency array **********
# In this exercise, you'll create a tf-idf word frequency array for a toy collection of documents. 
# For this, use the TfidfVectorizer from sklearn. It transforms a list of documents into a word frequency array, which it outputs as a csr_matrix. 
# It has fit() and transform() methods like other sklearn objects.

# You are given a list documents of toy documents about pets. Its contents have been printed in the IPython Shell.


# ********** Exercise Instructions **********
# 1 - Import TfidfVectorizer from sklearn.feature_extraction.text.
# 2 - Create a TfidfVectorizer instance called tfidf.
# 3 - Apply .fit_transform() method of tfidf to documents and assign the result to csr_mat. This is a word-frequency array in csr_matrix format.
# 4 - Inspect csr_mat by calling its .toarray() method and printing the result. This has been done for you.
# 5 - The columns of the array correspond to words. Get the list of words by calling the .get_feature_names() method of tfidf, and assign the result to words.


# ********** Script **********

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)


# ------------------------------------------------------------------------------

# ********** Clustering Wikipedia part I **********
# You saw in the video that TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format, such as word-frequency arrays. 
# Combine your knowledge of TruncatedSVD and k-means to cluster some popular pages from Wikipedia. 
# In this exercise, build the pipeline. In the next exercise, you'll apply it to the word-frequency array of some Wikipedia articles.

# Create a Pipeline object consisting of a TruncatedSVD followed by KMeans.
# (This time, we've precomputed the word-frequency matrix for you, so there's no need for a TfidfVectorizer).

# The Wikipedia dataset you will be working with was obtained from here.


# ********** Exercise Instructions **********
# 1 - Import:
# 1.1 - TruncatedSVD from sklearn.decomposition.
# 1.2 - KMeans from sklearn.cluster.
# 1.3 - make_pipeline from sklearn.pipeline.
# 2 - Create a TruncatedSVD instance called svd with n_components=50.
# 3 - Create a KMeans instance called kmeans with n_clusters=6.
# 4 - Create a pipeline called pipeline consisting of svd and kmeans.


# ********** Script **********

# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)


# ------------------------------------------------------------------------------

# ********** Clustering Wikipedia part II **********
# It is now time to put your pipeline from the previous exercise to work! 
# You are given an array articles of tf-idf word-frequencies of some popular Wikipedia articles, and a list titles of their titles. 
# Use your pipeline to cluster the Wikipedia articles.

# A solution to the previous exercise has been pre-loaded for you, so a Pipeline pipeline chaining TruncatedSVD with KMeans is available.


# ********** Exercise Instructions **********
# 1 - Import pandas as pd.
# 2 - Fit the pipeline to the word-frequency array articles.
# 3 - Predict the cluster labels.
# 4 - Align the cluster labels with the list titles of article titles by creating a DataFrame df with labels and titles as columns. This has been done for you.
# 5 - Use the .sort_values() method of df to sort the DataFrame by the 'label' column, and print the result.
# 6 - Hit 'Submit Answer' and take a moment to investigate your amazing clustering of Wikipedia pages!


# ********** Script **********

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))

