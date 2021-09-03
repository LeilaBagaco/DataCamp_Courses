# ********** NMF applied to Wikipedia articles **********
# In the video, you saw NMF applied to transform a toy word-frequency array. 
# Now it's your turn to apply NMF, this time using the tf-idf word-frequency array of Wikipedia articles, given as a csr matrix articles. 
# Here, fit the model and transform the articles. In the next exercise, you'll explore the result.


# ********** Exercise Instructions **********
# 1 - Import NMF from sklearn.decomposition.
# 2 - Create an NMF instance called model with 6 components.
# 3 - Fit the model to the word count data articles.
# 4 - Use the .transform() method of model to transform articles, and assign the result to nmf_features.
# 5 - Print nmf_features to get a first idea what it looks like (.round(2) rounds the entries to 2 decimal places.)


# ********** Script **********

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features.round(2))


# ------------------------------------------------------------------------------

# ********** NMF features of the Wikipedia articles **********
# Now you will explore the NMF features you created in the previous exercise. 
# A solution to the previous exercise has been pre-loaded, so the array nmf_features is available. 
# Also available is a list titles giving the title of each Wikipedia article.

# When investigating the features, notice that for both actors, the NMF feature 3 has by far the highest value. 
# This means that both articles are reconstructed using mainly the 3rd NMF component. In the next video, you'll see why: NMF components represent topics (for instance, acting!).


# ********** Exercise Instructions **********
# 1 - Import pandas as pd.
# 2 - Create a DataFrame df from nmf_features using pd.DataFrame(). Set the index to titles using index=titles.
# 3 - Use the .loc[] accessor of df to select the row with title 'Anne Hathaway', and print the result. These are the NMF features for the article about the actress Anne Hathaway.
# 4 - Repeat the last step for 'Denzel Washington' (another actor).


# ********** Script **********

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])


# ------------------------------------------------------------------------------

# ********** NMF learns topics of documents **********
# In the video, you learned when NMF is applied to documents, the components correspond to topics of documents, and the NMF features reconstruct the documents from the topics. 
# Verify this for yourself for the NMF model that you built earlier using the Wikipedia articles. 
# Previously, you saw that the 3rd NMF feature value was high for the articles about actors Anne Hathaway and Denzel Washington. 
# In this exercise, identify the topic of the corresponding NMF component.

# The NMF model you built earlier is available as model, while words is a list of the words that label the columns of the word-frequency array.

#After you are done, take a moment to recognise the topic that the articles about Anne Hathaway and Denzel Washington have in common!


# ********** Exercise Instructions **********
# 1 - Import pandas as pd.
# 2 - Create a DataFrame components_df from model.components_, setting columns=words so that columns are labeled by the words.
# 3 - Print components_df.shape to check the dimensions of the DataFrame.
# 4 - Use the .iloc[] accessor on the DataFrame components_df to select row 3. Assign the result to component.
# 5 - Call the .nlargest() method of component, and print the result. This gives the five words with the highest values for that component.


# ********** Script **********

# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())


# ------------------------------------------------------------------------------

# ********** Explore the LED digits dataset ********** 
# In the following exercises, you'll use NMF to decompose grayscale images into their commonly occurring patterns. 
# Firstly, explore the image dataset and see how it is encoded as an array. 
# You are given 100 images as a 2D array samples, where each row represents a single 13x8 image. 
# The images in your dataset are pictures of a LED digital display.


# ********** Exercise Instructions **********
# 1 - Import matplotlib.pyplot as plt.
# 2 - Select row 0 of samples and assign the result to digit. For example, to select column 2 of an array a, you could use a[:,2]. Remember that since samples is a NumPy array, you can't use the .loc[] or iloc[] accessors to select specific rows or columns.
# 3 - Print digit. This has been done for you. Notice that it is a 1D array of 0s and 1s.
# 4 - Use the .reshape() method of digit to get a 2D array with shape (13, 8). Assign the result to bitmap.
# 5 - Print bitmap, and notice that the 1s show the digit 7!
# 6 - Use the plt.imshow() function to display bitmap as an image.


# ********** Script **********

# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13, 8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


# ------------------------------------------------------------------------------

# ********** NMF learns the parts of images **********
# Now use what you've learned about NMF to decompose the digits dataset. You are again given the digit images as a 2D array samples. 
# This time, you are also provided with a function show_as_image() that displays the image encoded by any 1D array:
#def show_as_image(sample):
#    bitmap = sample.reshape((13, 8))
#    plt.figure()
#    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
#    plt.colorbar()
#    plt.show()

# After you are done, take a moment to look through the plots and notice how NMF has expressed the digit as a sum of the components!


# ********** Exercise Instructions **********
# 1 - Import NMF from sklearn.decomposition.
# 2 - Create an NMF instance called model with 7 components. (7 is the number of cells in an LED display).
# 3 - Apply the .fit_transform() method of model to samples. Assign the result to features.
# 4 - To each component of the model (accessed via model.components_), apply the show_as_image() function to that component inside the loop.
# 5 - Assign the row 0 of features to digit_features.
# 6 - Print digit_features.


# ********** Script **********

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)


# ------------------------------------------------------------------------------

# ********** PCA doesn't learn parts **********
# Unlike NMF, PCA doesn't learn the parts of things. Its components do not correspond to topics (in the case of documents) or to parts of images, when trained on images. 
# Verify this for yourself by inspecting the components of a PCA model fit to the dataset of LED digit images from the previous exercise. 
# The images are available as a 2D array samples. Also available is a modified version of the show_as_image() function which colors a pixel red if the value is negative.

# After submitting the answer, notice that the components of PCA do not represent meaningful parts of images of LED digits!


# ********** Exercise Instructions **********
# 1 - Import PCA from sklearn.decomposition.
# 2 - Create a PCA instance called model with 7 components.
# 3 - Apply the .fit_transform() method of model to samples. Assign the result to features.
# 4 - To each component of the model (accessed via model.components_), apply the show_as_image() function to that component inside the loop.


# ********** Script **********

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)


# ------------------------------------------------------------------------------
# ********** Which articles are similar to 'Cristiano Ronaldo'? **********
# In the video, you learned how to use NMF features and the cosine similarity to find similar articles. 
# Apply this to your NMF model for popular Wikipedia articles, by finding the articles most similar to the article about the footballer Cristiano Ronaldo. 
# The NMF features you obtained earlier are available as nmf_features, while titles is a list of the article titles.


# ********** Exercise Instructions **********
# 1 - Import normalize from sklearn.preprocessing.
# 2 - Apply the normalize() function to nmf_features. Store the result as norm_features.
# 3 - Create a DataFrame df from norm_features, using titles as an index.
# 4 - Use the .loc[] accessor of df to select the row of 'Cristiano Ronaldo'. Assign the result to article.
# 5 - Apply the .dot() method of df to article to calculate the cosine similarity of every row with article.
# 6 - Print the result of the .nlargest() method of similarities to display the most similiar articles. This has been done for you, so hit 'Submit Answer' to see the result!


# ********** Script **********

# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())


# ------------------------------------------------------------------------------

# ********** Recommend musical artists part I **********
# In this exercise and the next, you'll use what you've learned about NMF to recommend popular music artists! 
# You are given a sparse array artists whose rows correspond to artists and whose columns correspond to users. 
# The entries give the number of times each artist was listened to by each user.

# In this exercise, build a pipeline and transform the array into normalized NMF features. 
# The first step in the pipeline, MaxAbsScaler, transforms the data so that all users have the same influence on the model, regardless of how many different artists they've listened to. 
# In the next exercise, you'll use the resulting normalized NMF features for recommendation!


# ********** Exercise Instructions **********
# 1 - Import:
# 1.1 - NMF from sklearn.decomposition.
# 1.2 - Normalizer and MaxAbsScaler from sklearn.preprocessing.
# 1.3 - make_pipeline from sklearn.pipeline.
# 2 - Create an instance of MaxAbsScaler called scaler.
# 3 - Create an NMF instance with 20 components called nmf.
# 4 - Create an instance of Normalizer called normalizer.
# 5 - Create a pipeline called pipeline that chains together scaler, nmf, and normalizer.
# 6 - Apply the .fit_transform() method of pipeline to artists. Assign the result to norm_features.


# ********** Script **********

# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)


# ------------------------------------------------------------------------------

# ********** Recommend musical artists part II **********
# Suppose you were a big fan of Bruce Springsteen - which other musicial artists might you like? 
# Use your NMF features from the previous exercise and the cosine similarity to find similar musical artists. 
# A solution to the previous exercise has been run, so norm_features is an array containing the normalized NMF features as rows. 
# The names of the musical artists are available as the list artist_names.

# ********** Exercise Instructions **********
# 1 - Import pandas as pd.
# 2 - Create a DataFrame df from norm_features, using artist_names as an index.
# 3 - Use the .loc[] accessor of df to select the row of 'Bruce Springsteen'. Assign the result to artist.
# 4 - Apply the .dot() method of df to artist to calculate the dot product of every row with artist. Save the result as similarities.
#  5 - Print the result of the .nlargest() method of similarities to display the artists most similar to 'Bruce Springsteen'.


# ********** Script **********

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())

