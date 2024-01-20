import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load your dataset
training_path = '2018-Valence-oc-En-train.csv'
train_df = pd.read_csv(training_path)

nltk.download('wordnet')

# Preprocessing function (similar to before)
def preprocess_tweet(tweet):
    tokens = tweet.split()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Preprocess the tweets
train_df['Tweet'] = train_df['Tweet'].apply(preprocess_tweet)

# Tokenize the tweets
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_df['Tweet'])
sequences = tokenizer.texts_to_sequences(train_df['Tweet'])

# Pad the sequences
max_length = max(len(x) for x in sequences)
X = pad_sequences(sequences, maxlen=max_length)

# Encode the labels to categorical (one-hot encoding)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train_df['Intensity Class'])
y = to_categorical(integer_encoded)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=50, input_length=max_length))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))  # y_train.shape[1] should be the number of classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {accuracy:.2f}')

model.save('tweet_mental_state_model.h5')
model.save('tweet_mental_state_model.keras')