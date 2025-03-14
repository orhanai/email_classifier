import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential

# Parameters (adjust as needed)
vocab_size = 20000  # Should match the tokenizer's max_num_words
embedding_dim = 100  # You can choose this dimension based on your needs
max_sequence_length = 500  # Should match the padding length
num_classes = 20  # 20 Newsgroups

# Build the model
model = Sequential(
    [
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
        ),
        LSTM(128, return_sequences=False),  # Or try GRU, or multiple layers if needed
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    # loss="categorical_crossentropy",  #one-hot
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Display the model summary
summary = model.summary()
