import pandas as pd
import json
# These libraries are used for data manipulation and analysis.
# import contractions
import torch
import torch.nn as nn
#These libraries are used for building the deep learning model.
import tensorflow as tf
#This is another library used for building deep learning models.
from transformers import TFBertForSequenceClassification, BertTokenizer
#These are libraries specific to the BERT architecture, used for building and tokenizing the model.
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy
#These libraries are used for configuring the model's optimizer, loss function, callbacks, and metrics.
from sklearn.model_selection import train_test_split
#These libraries are used for data preprocessing and splitting the dataset into training and testing sets.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#This library is used for calculating the cosine similarity between two vectors.

df = pd.read_csv('booksummaries.txt',header=None,delimiter='\t')

#reads in a tab-separated text file of book summaries and creates a Pandas dataframe called df.



df.columns = ['Wikipedia article ID', 'Freebase ID', 'Book title', 'Author', 'Publication date', 'Book genres', 'Plot summary']

df = df.drop('Publication date', axis = 1)

#drops the "Publication date" column from the dataframe.
df.dropna()

#drops any rows with missing values from the dataframe, although it doesn't modify the original df dataframe. To modify df, need to assign the result of df.dropna() back to df
df['Plot summary'] = df['Plot summary'].apply(lambda x: str(x).lower())
df['Plot summary'] = df['Plot summary'].apply(lambda x: str(x).strip())
#apply string methods to convert all plot summaries to lowercase and remove any leading or trailing whitespace.
df.iloc[0]["Book genres"]

# retrieves the value in the "Book genres" column for the first row of the dataframe.

# df_10 = df.head(10)

df_10=df

'''
genre_dictionary that maps each book genre to its lowercase representation. It uses a Pandas dataframe called df_10 that presumably contains information on books 
and their genres. The code loops over each row of the dataframe and checks if the "Book genres" column contains any non-null values.
If it does, it parses the JSON string in that column to extract the genre information as a dictionary, 
and then adds each key-value pair in the dictionary to the genre_dictionary, 
with the key being the genre name and the value being the lowercase representation of the genre name.
The resulting genre_dictionary can be used as a lookup table to map genre names to their standardized lowercase representation.
'''
genre_dictionary = {}
for index, row in df_10.iterrows():
	if not pd.isna(row['Book genres']):
		genre_list = json.loads(row['Book genres'])

		for key, value in genre_list.items():
		    genre_dictionary[key] = value.lower()

print(genre_dictionary)

'''
genre_to_index that maps each genre in the genre_dictionary to a unique integer index.
starts by initializing an empty dictionary called genre_to_index and setting index to 0,
 then loops over all the genres in genre_dictionary.values(), which are the lowercase representations of the genres extracted from the book dataset
 . For each genre, it assigns a unique integer index to the genre_to_index dictionary and increments the index by 1.
 '''

genre_to_index = {}
index = 0

for genre in sorted(genre_dictionary.values()):
    genre_to_index[genre] = index
    index += 1

print(len(genre_to_index))


'''
index_to_genre that maps each integer index in the genre_to_index dictionary to its corresponding genre name.
 It does this by looping over each key-value pair in the genre_to_index dictionary using a dictionary comprehension. 
 For each key-value pair, it swaps the key-value order and adds the resulting key-value pair to a new dictionary called index_to_genre.
 This reverse mapping between integer indices and their corresponding genre names can be used to convert predicted genre labels back into their original string format.
'''

index_to_genre = {v: k for k, v in genre_to_index.items()}


'''
This function takes a dataFrame df containing book data and a dictionary genre_to_index that maps genres to integer indices, 
and returns a numpy array one_hot_orig containing one-hot encoded genre labels for each book in the dataset. 
The function begins by initializing an empty list called one_hot_orig. It then loops over each row in the DataFrame 
and extracts the genres associated with each book. For each book, it creates a genre vector of length len(genre_to_index) 
and sets the value at the index corresponding to the book's genre to 1, indicating that the book belongs to that genre. 
If a book does not have any genres listed in the dataset, it adds a zero vector to the one_hot_orig list. 
Finally, the function converts the one_hot_orig list to a numpy array and returns it. 
This one-hot encoding of genre labels is a common technique used to represent categorical data in machine learning models.

'''
def get_one_hot(df, genre_to_index):
    one_hot_orig = []
    for index, row in df.iterrows():
        if not pd.isna(row['Book genres']):
            genre_list = json.loads(row['Book genres'])
            genre_vector = [0] * len(genre_to_index)
            for key, value in genre_list.items():
                genre = value.lower()
                genre_vector[genre_to_index[genre]] = 1
            one_hot_orig.append(genre_vector)
        else:
            one_hot_orig.append([0] * len(genre_to_index))
    return np.array(one_hot_orig)


onehot_vectors = get_one_hot(df_10,genre_to_index)   

# Load the pre-trained BERT model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(genre_to_index))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare the training data
train_texts = df_10["Plot summary"].tolist()
train_labels = onehot_vectors # one-hot encoded labels for each genre
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

# Split the data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_encodings['input_ids'], train_labels, test_size=0.2)
# train_inputs = train_texts[:8]
# val_inputs = train_texts[8:10]
# train_labels = onehot_vectors[:8]
# val_labels = onehot_vectors[8:10]


print(train_encodings.keys())


# Create TensorFlow datasets from the training and validation data
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
train_dataset = train_dataset.batch(16).shuffle(100)

val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels))
val_dataset = val_dataset.batch(16)

print(train_dataset)

'''
it initializes an instance of the Adam optimizer with a learning rate of 2e-5 and an epsilon value of 1e-8. 
The Adam optimizer is an algorithm for optimizing gradient descent in neural networks. 
The from_logits parameter of the BinaryCrossentropy loss function is set to True, which indicates that the model will output raw logits rather than probabilities. 
The BinaryCrossentropy loss function is commonly used for binary classification problems, where each sample belongs to one of two classes. 
The early_stopping callback monitors the validation loss during training and stops training if the validation loss does not improve after three epochs. 
The BinaryAccuracy metric is used to evaluate the performance of the model during training and validation by computing the accuracy of the predicted labels 
with respect to the true labels.
'''
# Set up the optimizer and training parameters
optimizer = Adam(learning_rate=2e-5, epsilon=1e-8)
loss_fct = BinaryCrossentropy(from_logits=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
binary_accuracy = BinaryAccuracy()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fct, metrics=[binary_accuracy])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping])

# Select 10-15 records from df for prediction
test_df = df.iloc[0:10]

'''
extracts the plot summaries from the test dataframe as a list of strings.
The tokenizer function is then used to tokenize the test texts, truncating them to a maximum length of 512 tokens and padding them to ensure that all
inputs have the same length. The resulting encoded inputs are then used to create a tf.data.Dataset object using the from_tensor_slices method,
which slices the input arrays along their first dimension to produce a sequence of elements, each consisting of a pair of input and attention mask arrays.
The resulting dataset is then batched with a batch size of 16, which means that the input data will be processed in batches of 16 samples at a time during evaluation
'''
# Preprocess the test data
test_texts = test_df["Plot summary"].tolist()
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
test_dataset = tf.data.Dataset.from_tensor_slices((test_encodings['input_ids'], test_encodings['attention_mask']))
test_dataset = test_dataset.batch(16)

# Make predictions on the test data
predictions = model.predict(test_dataset)

predictions_array = predictions.logits
one_hot_pred = np.zeros(predictions_array.shape, dtype=int)

for i in range(predictions_array.shape[0]):
    for j in range(predictions_array.shape[1]):
        if predictions_array[i][j] > 0.15:
            one_hot_pred[i][j] = 1


print(one_hot_pred[0])

print(predictions_array[0])

test_df.iloc[0]['Book genres']

one_hot_orig = get_one_hot(test_df,genre_to_index)

print(one_hot_orig[0])

'''
compares each element of the orig and pred lists and checks how many genres match.
If the number of matching genres is greater than or equal to the threshold, then the prediction is considered accurate.
The function then computes the overall accuracy by dividing the total number of accurate predictions by the length of the orig list. 
The resulting accuracy is returned as a float.
'''
def check_similarity(orig, pred, threshold, genre_count):
    # Check if lengths of orig and pred are equal
    if len(orig) != len(pred):
        return False
    total_acc = 0
    # Compute the number of elements that match in orig and pred
    for i in range(len(orig)):
      match_count = 0
      for j in range(len(orig[i])):
        if orig[i][j] == pred[i][j]:
            match_count += 1
      if match_count>threshold:
        total_acc+=1
    return total_acc/len(orig)

accuracy = check_similarity(one_hot_orig,one_hot_pred,5,len(genre_to_index))*100

print(accuracy) #Accrracy from the original description using the BERT classifier is 58%.

# assuming one_hot_orig and one_hot_pred are numpy arrays with shape (num_samples, num_classes)
cosine_sims = cosine_similarity(one_hot_orig, one_hot_pred)
accurate_preds = (cosine_sims.diagonal() >= 0.1).sum()
cosine_sim = accurate_preds / len(one_hot_orig)

print(cosine_sim)


print(one_hot_orig)

print(one_hot_pred)


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import json
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy
from sklearn.model_selection import train_test_split
import numpy as np

# reads in a tab-separated values file ('booksummaries.txt') into a Pandas DataFrame, and assigns column names to the DataFrame, with delimiter='\t' parameter specifies that the values in the file are separated by tabs.
df = pd.read_csv('booksummaries.txt',header=None,delimiter='\t')
df.columns = ['Wikipedia article ID', 'Freebase ID', 'Book title', 'Author', 'Publication date', 'Book genres', 'Plot summary']

#dropping of publication date column from data
df = df.drop('Publication date', axis = 1)
print(df.shape)
df = df.dropna()
print(df.shape)

#The <= 4 condition checks whether the length of the list of genres associated with each book is less than or equal to 4. Finally, the resulting boolean mask is used to filter the rows of the DataFrame df using boolean indexing. The filtered DataFrame is then assigned back to the variable df.
df=df[df['Book genres'].apply(lambda x: len(eval(x))) <= 4]

#This code filters the DataFrame df by selecting only the rows where the length of the plot summary associated with each book is between 1500 and 4000 characters long.
df = df[df['Plot summary'].apply(lambda x: len(x)) <= 4000]
df = df[df['Plot summary'].apply(lambda x: len(x)) >= 1500]

df.shape
df = df.head(2000)

#resetting the index
df = df.reset_index(drop=True)

#reading the csv file df_pre_summary
df.to_csv("df_pre_summary.csv", index=False)
test_df = pd.read_csv("df_pre_summary.csv")

#Using the GPU to perform computations faster
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#This code imports the pipeline module from the Hugging Face transformers library and creates a BART (Bidirectional and Auto-Regressive Transformer) summarization pipeline using the Facebook BART-Large-CNN pre-trained model.
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Iterate through the rows of the DataFrame
# The code then generates a summary using the summarizer pipeline with the specified parameters, including max_length=ml and min_length=30, which sets the minimum length of the generated summary to 30 characters. The do_sample=True parameter enables the model to generate multiple summary candidates, allowing it to produce more diverse summaries.
count = 1
for index, row in test_df.iterrows():
    plot_summary = row['Plot summary']
    ml = 200
    if len(plot_summary) < 300:
        print('index'+str(index))
        print(plot_summary)
        ml = len(plot_summary)/2
    summary = summarizer(plot_summary, max_length=ml, min_length=30, do_sample=True)[0]['summary_text']
    test_df.loc[index, 'gen_summary'] = summary
    if index%10==0:
        print("Record: "+str(index+1)+" | Original length: "+str(len(plot_summary)) + " | " + "Summary length: " + str(len(summary)))

#saving the df_facebook_summary csv file
test_df.to_csv("df_facebook_summary.csv", index=False)

#printing the gen_summary
test_df.loc[673, 'gen_summary']

fb_df = test_df


# Import necessary libraries and packages
import pandas as pd
import json
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Load the Pegasus tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")

# Load the test and train dataframes
test_df = pd.read_csv("test_df.csv")
train_df = pd.read_csv("train_df.csv")

# Define a function to generate summaries using the Pegasus model
def generate_summary(text):
    # Tokenize the text using the Pegasus tokenizer
    input_ids = tokenizer.encode(text, return_tensors="pt")
    # Generate the summary using the Pegasus model
    summary_ids = model.generate(input_ids)
    # Decode the summary and remove any special tokens
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Filter the test dataframe to include only rows with plot summaries of at least 300 characters
filtered_test_df = test_df[test_df['Plot summary'].str.len() >= 300]

# Iterate through the filtered test dataframe and generate summaries for each plot summary
for index, row in filtered_test_df.iterrows():
    plot_summary = row['Plot summary']
    max_length = 4000
    print('record: ' + str((index+1)), end=' | ')
    print('orig length: ' + str(len(plot_summary)), end=' | ')
    # If the plot summary is longer than the maximum length, truncate it
    if len(plot_summary) > max_length:
        plot_summary = plot_summary[:max_length]
    # Generate the summary using the Pegasus model
    summary = generate_summary(plot_summary)
    # Add the generated summary to the dataframe
    filtered_test_df.loc[index, 'gen_summary'] = summary
    print('summary length: ' + str(len(summary)))

# Save the filtered test dataframe with the generated summaries to a CSV file
filtered_test_df.to_csv("df_google_summary.csv", index=False)

# Print the generated summary for a specific row in the test dataframe
print(test_df.loc[673, 'gen_summary'])

gg_df = filtered_test_df


# LSTM Reference: https://www.kaggle.com/code/mohamedf000/seq2seq-enc-dec
!pip install rouge

import numpy as np
import pandas as pd

import re
import string
import csv
from sklearn.model_selection import train_test_split
import contractions
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Concatenate, TimeDistributed, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from rouge import Rouge

train_data = pd.read_csv('df_facebook_summary.csv')
train_data.head()

train_data = train_data.drop('id', axis=1)
train_data = train_data.reset_index(drop=True)
# test_data = test_data.drop(['id'], axis=1)
# test_data = test_data.reset_index(drop=True)

def clean_text(text, remove_stopwords=True):
    # convert text to lowercase
    text = text.lower()
    # replace contractions with their expanded forms
    text = contractions.fix(text)
    # remove URLs
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # remove HTML tags
    text = re.sub(r'\<.*?\>', ' ', text)
    # remove special characters
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    # replace line breaks with spaces
    text = re.sub(r'\n', ' ', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_stopwords:
        # remove stopwords
        stops = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stops])
        
    return text


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# This code is creating a list of clean summaries by applying the clean_text() function to each summary in the train_data.highlights list. The clean_text() 
#function is called with the remove_stopwords parameter set to False, which means that stopwords will not be removed from the summaries. The resulting list 
#of clean summaries is stored in the clean_summaries variable. Finally, a message is printed to indicate that the cleaning of the summaries is complete.
clean_summaries = []
for summary in train_data.highlights:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print('Cleaning Summaries Complete')
   
#This code is cleaning the text data from the gen_summary column of the train_data DataFrame using the clean_text() function and storing the cleaned 
#text in the clean_texts list. It also prints a message indicating that the cleaning process is complete. Finally, it deletes the train_data object from 
#memory to free up space.    
clean_texts = []
for text in train_data.gen_summary:
    clean_texts.append(clean_text(text))
print('Cleaning Texts Complete')
del train_data

#This code creates a new Pandas DataFrame clean_df containing preprocessed texts and summaries. The first 3000 preprocessed texts and 
#summaries are added as columns 'text' and 'summary' respectively. Any empty summaries are replaced with NaN values, and then any rows with NaN values are dropped.
clean_df = pd.DataFrame()
clean_df['text'] = clean_texts[:3000]
clean_df['summary'] = clean_summaries[:3000]
clean_df['summary'].replace('', np.nan, inplace=True)
clean_df.dropna(axis=0, inplace=True)

#The code is adding special start and end tokens, '<sostok>' and '<eostok>', respectively, to each summary in the 'summary' 
#column of the 'clean_df' dataframe. Then, it deletes the 'clean_texts' and 'clean_summaries' lists from memory.
clean_df['summary'] = clean_df['summary'].apply(lambda x: '<sostok>' + ' ' + x + ' ' + '<eostok>')
del clean_texts
del clean_summaries

#This code is splitting the cleaned data into training and testing sets, with 10% of the data being used for testing and 90% for training. 
#The training set consists of the cleaned text data and corresponding summaries, which are stored in train_x and train_y, respectively. 
#The testing set consists of a similar split of the cleaned data, with the text data and summaries stored in test_x and test_y, respectively. 
#The original cleaned data is then deleted from memory.
train_x, test_x, train_y, test_y = train_test_split(clean_df['text'], clean_df['summary'], test_size=0.1, random_state=0)
del clean_df

#This code initializes a new tokenizer object t_tokenizer and fits it on the list of train_x texts using the fit_on_texts() method. 
#This step is necessary to create a vocabulary from the training data that will be used to convert the text into numerical sequences.
t_tokenizer = Tokenizer()
t_tokenizer.fit_on_texts(list(train_x))

#This code is calculating the number of words in the Tokenizer's vocabulary that appear fewer than thresh number of times in the training data, 
#as well as the total number of words and their frequencies. It does so by iterating over the word_counts dictionary of the t_tokenizer object 
#and incrementing count and frequency variables if a word appears less than thresh times. The total_count and total_frequency variables are incremented 
#for every word in the vocabulary, regardless of their frequency.
thresh = 4
count = 0
total_count = 0
frequency = 0
total_frequency = 0

for key, value in t_tokenizer.word_counts.items():
    total_count += 1
    total_frequency += value
    if value < thresh:
        count += 1
        frequency += value


print('% of rare words in vocabulary: ', (count/total_count)*100.0)
print('Total Coverage of rare words: ', (frequency/total_frequency)*100.0)
t_max_features = total_count - count
print('Text Vocab: ', t_max_features)

s_tokenizer = Tokenizer()
s_tokenizer.fit_on_texts(list(train_y))

thresh = 6
count = 0
total_count = 0
frequency = 0
total_frequency = 0

for key, value in s_tokenizer.word_counts.items():
    total_count += 1
    total_frequency += value
    if value < thresh:
        count += 1
        frequency += value

print('% of rare words in vocabulary: ', (count/total_count)*100.0)
print('Total Coverage of rare words: ', (frequency/total_frequency)*100.0)
s_max_features = total_count-count
print('Summary Vocab: ', s_max_features)

maxlen_text = 800
maxlen_summ = 150

#This code prepares the training and validation data for sequence-to-sequence model by converting the text input sequences into sequences of integers 
#using the Tokenizer class from Keras. It also sets the maximum number of words in the vocabulary to t_max_features using the num_words argument of the 
#Tokenizer class constructor. The resulting sequences are stored in train_x and val_x, which will be used as inputs for the model during training and validation, respectively.
val_x = test_x
t_tokenizer = Tokenizer(num_words=t_max_features)
t_tokenizer.fit_on_texts(list(train_x))
train_x = t_tokenizer.texts_to_sequences(train_x)
val_x = t_tokenizer.texts_to_sequences(val_x)

#This code is padding the sequences in train_x and val_x to make sure that all sequences have the same length maxlen_text. Padding is added to the end of the sequences 
#using the padding='post' parameter. This is typically done to feed the data into a neural network, where sequences of different lengths cannot be processed together.
train_x = pad_sequences(train_x, maxlen=maxlen_text, padding='post')
val_x = pad_sequences(val_x, maxlen=maxlen_text, padding='post')

#This code initializes a Tokenizer object for the summary texts and sets the maximum number of words to be included in the vocabulary using the num_words parameter. 
#The fit_on_texts method is then called to update the tokenizer's internal vocabulary based on the training data. After that, texts_to_sequences method is used to convert 
#the summary texts in both the training and validation set into sequences of integers using the learned vocabulary. The training set is updated to hold these integer sequences. 
#The same steps are also applied to the validation set (val_y) to ensure that the same tokenizer and vocabulary are used for both sets.
val_y = test_y
s_tokenizer = Tokenizer(num_words=s_max_features)
s_tokenizer.fit_on_texts(list(train_y))
train_y = s_tokenizer.texts_to_sequences(train_y)
val_y = s_tokenizer.texts_to_sequences(val_y)

train_y = pad_sequences(train_y, maxlen=maxlen_summ, padding='post')
val_y = pad_sequences(val_y, maxlen=maxlen_summ, padding='post')

print("Training Sequence", train_x.shape)
print('Target Values Shape', train_y.shape)
print('Test Sequence', val_x.shape)
print('Target Test Shape', val_y.shape)

# Initialize an empty dictionary to store the word embeddings
embeding_index = {}

# Set the embedding dimension to 100
embed_dim = 100

# Open the file containing the pre-trained GloVe word embeddings
with open('glove.6B.100d') as f:
    # Iterate through each line of the file
    for line in f:
        # Split each line into a list of values
        values = line.split()
        # The first value is the word
        word = values[0]
        # The remaining values are the embedding coefficients
        coefs = np.asarray(values[1:], dtype='float32')
        # Add the word and its embedding coefficients to the dictionary
        embeding_index[word] = coefs

# Initialize a matrix of zeros to hold the word embeddings for the input text
t_embed = np.zeros((t_max_features, embed_dim))
# Iterate through each word in the input text's tokenizer word index
for word, i in t_tokenizer.word_index.items():
    # Get the word's embedding coefficients from the dictionary, if it exists
    vec = embeding_index.get(word)
    # If the word's index is less than the maximum number of words allowed and it has a non-null embedding,
    # add the embedding coefficients to the embedding matrix
    if i < t_max_features and vec is not None:
        t_embed[i] = vec

# Initialize a matrix of zeros to hold the word embeddings for the output summary
s_embed = np.zeros((s_max_features, embed_dim))
# Iterate through each word in the output summary's tokenizer word index
for word, i in s_tokenizer.word_index.items():
    # Get the word's embedding coefficients from the dictionary, if it exists
    vec = embeding_index.get(word)
    # If the word's index is less than the maximum number of words allowed and it has a non-null embedding,
    # add the embedding coefficients to the embedding matrix
    if i < s_max_features and vec is not None:
        s_embed[i] = vec

# Delete the embedding index dictionary to free up memory
del embeding_index


# Set the dimensionality of the latent space
latent_dim = 128

# Define the encoder input layer
enc_input = Input(shape=(maxlen_text, ))

# Define the embedding layer for the encoder
enc_embed = Embedding(t_max_features, embed_dim, input_length=maxlen_text, weights=[t_embed], trainable=False)(enc_input)

# Define the Bidirectional LSTM layer for the encoder
enc_lstm = Bidirectional(LSTM(latent_dim, return_state=True))

# Connect the encoder input layer to the embedding layer and then to the Bidirectional LSTM layer
enc_output, enc_fh, enc_fc, enc_bh, enc_bc = enc_lstm(enc_embed)

# Concatenate the forward and backward hidden states of the encoder LSTM to obtain the final hidden state
enc_h = Concatenate(axis=-1, name='enc_h')([enc_fh, enc_bh])

# Concatenate the forward and backward cell states of the encoder LSTM to obtain the final cell state
enc_c = Concatenate(axis=-1, name='enc_c')([enc_fc, enc_bc])

# Define the decoder input layer
dec_input = Input(shape=(None, ))

# Define the embedding layer for the decoder
dec_embed = Embedding(s_max_features, embed_dim, weights=[s_embed], trainable=False)(dec_input)

# Define the LSTM layer for the decoder, with a dropout rate of 0.3 and recurrent dropout rate of 0.2
dec_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)

# Connect the decoder embedding layer to the LSTM layer, with the final encoder hidden and cell states as initial states
dec_outputs, _, _ = dec_lstm(dec_embed, initial_state=[enc_h, enc_c])

# Define the TimeDistributed dense layer for the decoder, with softmax activation
dec_dense = TimeDistributed(Dense(s_max_features, activation='softmax'))

# Connect the output of the decoder LSTM to the TimeDistributed dense layer
dec_output = dec_dense(dec_outputs)

# Define the model with the encoder and decoder inputs and the decoder output
model = Model([enc_input, dec_input], dec_output)

# Print a summary of the model architecture
model.summary()


# Plot the model architecture
plot_model(
    model,
    to_file='./seq2seq_encoder_decoder.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96)

# Compile the model with sparse categorical cross-entropy loss and RMSprop optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

# Define early stopping callback to stop training if validation loss stops improving
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

# Train the model on the training set with early stopping and batch size of 128 for 20 epochs
model.fit([train_x, train_y[:, :-1]], train_y.reshape(train_y.shape[0], train_y.shape[1], 1)[:, 1:], 
          epochs=20, callbacks=[early_stop], batch_size=128, verbose=1, 
          validation_data=([val_x, val_y[:, :-1]], val_y.reshape(val_y.shape[0], val_y.shape[1], 1)[:, 1:]))

# Save the trained model
model.save('lstm_10000_records.h5')

#The code imports the load_model function from the Keras module of TensorFlow and uses it to load a previously saved model from a file named 'lstm_10000_records.h5'. 
#It then reads in two CSV files named 'test_df.csv' and 'train_df.csv' using the pandas library.
from tensorflow.keras.models import load_model
loaded_model = load_model('lstm_10000_records.h5')

test_df = pd.read_csv("test_df.csv")
train_df = pd.read_csv("train_df.csv")

#The code loads the saved LSTM model from the 'lstm_10000_records.h5' file and defines a function 'generate_summary' which takes a text input, 
#converts it to a padded sequence using the text tokenizer, and generates a summary using the loaded LSTM model and the summary tokenizer. 
#The code then loads a test dataframe 'test_df' and applies the 'generate_summary' function to the 'Plot summary' column of the first row of the dataframe. 
#Finally, the code reads a CSV file 'df_google_summary.csv' and creates a new dataframe 'lstm_pred_df' which will be used to store the generated summaries.
from tensorflow.keras.models import load_model
lstm_model = load_model('lstm_10000_records.h5')

def generate_summary(text):
    text_seq = t_tokenizer.texts_to_sequences([text])
    padded_text_seq = pad_sequences(text_seq, maxlen=maxlen_text, padding='post')
    summary_seq = lstm_model.predict([padded_text_seq, np.zeros((1, 1))])
    summary = s_tokenizer.sequences_to_texts(summary_seq)[0]
    return summary

test_df.iloc[0]['Plot summary']

generate_summary(test_df.iloc[0]['Plot summary'])

test_df = pd.read_csv("df_google_summary.csv")  

lstm_pred_df = test_df


#Checking cosine similarities of summaries:
#1) LSTM predicted summary with Facebook (BART) generated summary.
#2) LSTM predicted summary with Google (Pegasus) generated summary.
one_hot_orig_fb = get_one_hot(fb_df,genre_to_index)
one_hot_orig_gg = get_one_hot(gg_df,genre_to_index)
one_hot_orig_lstm = get_one_hot(lstm_pred_df,genre_to_index)


cosine_sims = cosine_similarity(one_hot_orig_fb, one_hot_orig_lstm) 
accurate_preds = (cosine_sims.diagonal() >= 0.1).sum()
cosine_sim_fb = accurate_preds / len(one_hot_orig)

print(cosine_sim_fb) #Cosine similarity between the LSTM predicted summary and the BART generated summary is 0.31.

cosine_sims = cosine_similarity(one_hot_orig_gg, one_hot_orig_lstm)
accurate_preds = (cosine_sims.diagonal() >= 0.1).sum() 
cosine_sim_gg = accurate_preds / len(one_hot_orig)

print(cosine_sim_gg) #Cosine similarity between the LSTM predicted summary and the Google Pegasus generated summary is 0.23.



import pandas as pd
import json
# import contractions
import torch
import torch.nn as nn
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

'''This block imports necessary libraries and functions that will be used later in the code.
pandas is a data manipulation library, json is used to work with JSON data format, torch is a machine learning library, and tensorflow is another machine learning library.
TFBertForSequenceClassification and BertTokenizer are classes from the transformers library used to build and preprocess the model respectively.
Adam is an optimization algorithm used for training the model, BinaryCrossentropy is a loss function, and BinaryAccuracy is a metric used to evaluate the performance of the model.
train_test_split is used to split the dataset into training and validation sets.
numpy is a numerical computing library and cosine_similarity is a function used to measure the similarity between two vectors.'''

df_10 = lstm_pred_df

'''
This block creates a dictionary of book genres extracted from a dataframe lstm_pred_df.
The dataframe is assigned to a new variable df_10.
The genre_dictionary is initialized as an empty dictionary.
The iterrows() function is used to iterate over each row in the dataframe.
The if condition checks if the 'Book genres' column in the current row is not null. If the condition is satisfied, the genre_list is loaded from the 'Book genres' column using json.loads().
A for loop is used to iterate over each key-value pair in the genre_list dictionary.
The key-value pair is added to the genre_dictionary with the key as-is and the value converted to lowercase using the lower() method.'''

genre_dictionary = {}
for index, row in df_10.iterrows():
    if not pd.isna(row['Book genres']):
        genre_list = json.loads(row['Book genres'])

        for key, value in genre_list.items():
            genre_dictionary[key] = value.lower()

print(genre_dictionary)

'''
This block creates a mapping of book genres to their respective indices.
A new dictionary genre_to_index is initialized as an empty dictionary.
A variable index is initialized with a value of zero.
The for loop iterates over each value in the genre_dictionary dictionary, which contains the lowercased book genres.
The sorted() function sorts the values in the dictionary in ascending order and returns them as a list.
For each genre in the sorted list, a new key-value pair is added to the genre_to_index dictionary, where the key is the genre and the value is the current value of the index variable.
After adding the key-value pair, the index variable is incremented by 1 to keep track of the current index value.'''

genre_to_index = {}
index = 0

for genre in sorted(genre_dictionary.values()):
    genre_to_index[genre] = index
    index += 1

print(len(genre_to_index))
''''
The code creates a function get_one_hot() that takes a dataframe and a dictionary that maps book genres to indices. 
It then iterates over each row in the dataframe, and for each row, it creates a one-hot encoding of the book genres using the dictionary. 
If the 'Book genres' column in the current row is null, the function returns a list of zeros with the length equal to the number of book 
genres in the dictionary. Otherwise, it updates a genre_vector list with ones at the indices corresponding to the book genres, and appends 
this list to a one_hot_orig list. Finally, the one_hot_orig list is converted to a NumPy array and returned.'''

index_to_genre = {v: k for k, v in genre_to_index.items()}

def get_one_hot(df, genre_to_index):
    one_hot_orig = []
    for index, row in df.iterrows():
        if not pd.isna(row['Book genres']):
            genre_list = json.loads(row['Book genres'])
            genre_vector = [0] * len(genre_to_index)
            for key, value in genre_list.items():
                genre = value.lower()
                genre_vector[genre_to_index[genre]] = 1
            one_hot_orig.append(genre_vector)
        else:
            one_hot_orig.append([0] * len(genre_to_index))
    return np.array(one_hot_orig)



'''
onehot_vectors = get_one_hot(df_10,genre_to_index) creates a one-hot encoding of the book genres in the given DataFrame df_10 using 
the get_one_hot() function and the genre_to_index dictionary.

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(genre_to_index)) loads the pre-trained 
BERT model with the "bert-base-uncased" architecture for sequence classification, and sets the number of labels to be equal to the 
number of book genres in the genre_to_index dictionary.

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") loads the pre-trained tokenizer corresponding to the "bert-base-uncased" 
BERT model.'''

onehot_vectors = get_one_hot(df_10,genre_to_index)   

# Load the pre-trained BERT model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(genre_to_index))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


'''
The first line of code extracts plot summaries from a DataFrame and converts them to a list of strings. 
The second line assigns one-hot encoded genre labels to a variable. The third line tokenizes the plot summaries using a pre-trained 
tokenizer, pads/truncates the sequences to a maximum length of 512 tokens, and returns a dictionary containing the tokenized input 
sequences and attention masks. The fourth line splits the data into training and validation sets, where 80% is used for training and 
20% is used for validation. The token IDs of the input sequences are stored in the train_inputs and val_inputs variables, while the 
corresponding genre labels are stored in train_labels and val_labels.
'''
# Prepare the training data
train_texts = df_10["Plot summary"].tolist()
train_labels = onehot_vectors # one-hot encoded labels for each genre
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

# Split the data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_encodings['input_ids'], train_labels, test_size=0.2)
# train_inputs = train_texts[:8]
# val_inputs = train_texts[8:10]
# train_labels = onehot_vectors[:8]
# val_labels = onehot_vectors[8:10]


print(train_encodings.keys())

'''
The first two lines of code create TensorFlow datasets from the input and label data, where each data point in the dataset is a pair of 
input sequence and corresponding label. The data is sliced into batches of 16 data points and shuffled randomly with a buffer size of 
100 for the training dataset. The validation dataset is also sliced into batches of 16 data points but is not shuffled.
'''

# Create TensorFlow datasets from the training and validation data
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
train_dataset = train_dataset.batch(16).shuffle(100)

val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels))
val_dataset = val_dataset.batch(16)

print(train_dataset)


'''
The first line creates an instance of the Adam optimizer with a learning rate of 2e-5 and an epsilon value of 1e-8. 
The second line creates an instance of the binary cross-entropy loss function to calculate the loss between predicted and 
actual genre labels. The third line sets up early stopping to monitor the validation loss and stop the model training if the 
validation loss does not improve for 3 epochs. The fourth line creates an instance of the binary accuracy metric to evaluate 
the model's accuracy during training and validation.
'''
# Set up the optimizer and training parameters
optimizer = Adam(learning_rate=2e-5, epsilon=1e-8)
loss_fct = BinaryCrossentropy(from_logits=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
binary_accuracy = BinaryAccuracy()


'''
compiles the model using the Adam optimizer, binary cross-entropy loss function, and binary accuracy metric. 
The second line trains the model using the training dataset for 10 epochs and validates it using the validation dataset, 
while also stopping early if the validation loss does not improve for 3 epochs. The third line selects 10 records from the 
original dataframe for prediction.
'''
# Compile the model
model.compile(optimizer=optimizer, loss=loss_fct, metrics=[binary_accuracy])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping])

# Select 10-15 records from df for prediction
test_df = df.iloc[0:10]

'''
The code is preparing the test dataset by encoding the text summaries of 10 records from the original dataframe using the BERT tokenizer. 
The encoded text and attention mask are then converted into a TensorFlow dataset object and split into batches of size 16 for prediction.
'''

# Preprocess the test data
test_texts = test_df["Plot summary"].tolist()
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
test_dataset = tf.data.Dataset.from_tensor_slices((test_encodings['input_ids'], test_encodings['attention_mask']))
test_dataset = test_dataset.batch(16)

'''
The above code predicts the genres for the test dataset using the trained model. The predictions are initially in the form of a logits array. 
The code then converts this array to a one-hot-encoded array using a threshold value of 0.15. If the value in the logits array is greater 
than 0.15, the corresponding element in the one-hot-encoded array is set to 1, otherwise, it is set to 0.
'''
# Make predictions on the test data
predictions = model.predict(test_dataset)

predictions_array = predictions.logits
one_hot_pred = np.zeros(predictions_array.shape, dtype=int)

for i in range(predictions_array.shape[0]):
    for j in range(predictions_array.shape[1]):
        if predictions_array[i][j] > 0.15:
            one_hot_pred[i][j] = 1


print(one_hot_pred[0])

print(predictions_array[0])


'''
define a function that computes the similarity between the original one-hot encoded labels and the predicted one-hot encoded labels. 
The function takes in four arguments - the original one-hot encoded labels (orig), the predicted one-hot encoded labels (pred), 
a threshold value for similarity, and the number of genres in the dataset (genre_count). The function computes the number of matching 
elements between the original and predicted vectors for each record, and returns the percentage of records that have a match count 
greater than the threshold.
'''
test_df.iloc[0]['Book genres']

one_hot_orig = get_one_hot(test_df,genre_to_index)

print(one_hot_orig[0])

def check_similarity(orig, pred, threshold, genre_count):
    # Check if lengths of orig and pred are equal
    if len(orig) != len(pred):
        return False
    total_acc = 0
    # Compute the number of elements that match in orig and pred
    for i in range(len(orig)):
      match_count = 0
      for j in range(len(orig[i])):
        if orig[i][j] == pred[i][j]:
            match_count += 1
      if match_count>threshold:
        total_acc+=1
    return total_acc/len(orig)

accuracy = check_similarity(one_hot_orig,one_hot_pred,5,len(genre_to_index))*100

print(accuracy) #Accrracy from the summarized description (LSTM) using the BERT classifier is 27%.

'''
The code computes cosine similarity between the one-hot encoded original and predicted genre vectors, and checks if the similarity 
score is above a certain threshold. It counts the number of accurate predictions where the cosine similarity score is above the 
threshold, and calculates the overall accuracy by dividing the accurate predictions by the total number of predictions.
'''
# assuming one_hot_orig and one_hot_pred are numpy arrays with shape (num_samples, num_classes)
# cosine_sims = cosine_similarity(one_hot_orig, one_hot_pred)
# accurate_preds = (cosine_sims.diagonal() >= 0.1).sum()
# cosine_sim = accurate_preds / len(one_hot_orig)

# print(cosine_sim)


# print(one_hot_orig)

# print(one_hot_pred)