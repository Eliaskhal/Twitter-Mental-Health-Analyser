import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the Tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Load the maxlen variable
with open('maxlen.pkl', 'rb') as maxlen_file:
    maxlen = pickle.load(maxlen_file)

# Load the trained model
model = load_model('tweet_mental_state_model.keras') 

while True:
    # Test a single tweet
    test_tweet = input('> ')
    sequence = tokenizer.texts_to_sequences([test_tweet])
    X_single_test = pad_sequences(sequence, maxlen=maxlen)

    # Make predictions on the single test tweet
    prediction = model.predict(X_single_test)
    predicted_class = label_encoder.inverse_transform(prediction.argmax(axis=1))[0]

    # Print the result
    print(f'Predicted Intensity Class: {predicted_class}')
