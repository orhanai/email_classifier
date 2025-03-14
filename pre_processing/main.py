import logging
import re

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Uncomment these lines if you plan to use stopword removal and haven't downloaded NLTK stopwords yet
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# STOPWORDS = set(stopwords.words('english'))
# or use generic
STOPWORDS = {"the", "is", "in", "and", "to", "of"}

SUBSET_SIZE = 1000


def clean_text(text: str) -> str:
    """
    Clean the input text by lowercasing, removing special characters, and extra whitespace.
    """
    # Lowercase the text
    text = text.lower()
    # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_preprocess_data(subset: str = "all"):
    """
    Load the dataset and preprocess the texts.

    Returns:
        texts: List of cleaned text data.
        labels: Corresponding labels.
    """
    logging.info("Loading the dataset (%s)...", subset)
    data = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))

    texts = [clean_text(text) for text in data.data[:SUBSET_SIZE]]
    labels = data.target[:SUBSET_SIZE]
    logging.info("Loaded %d documents.", len(texts))
    return texts, labels


def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from the input text.
    Note: This function is defined for future use and is not currently integrated into the pipeline.

    Args:
        text: Input string.

    Returns:
        Text string without stopwords.
    """
    # Split the text into words and filter out stopwords
    filtered_words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(filtered_words)


def tokenize_and_pad(texts, max_num_words: int = 20000, max_sequence_length: int = 500):
    """
    Tokenize the texts and pad the sequences to a fixed length.

    Args:
        texts: List of text documents.
        max_num_words: Maximum number of words to keep based on frequency.
        max_sequence_length: Maximum length of all sequences.

    Returns:
        sequences_padded: Padded sequences.
        word_index: Dictionary mapping words to their indices.
    """
    logging.info("Tokenizing texts...")
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    logging.info("Padding sequences to a length of %d...", max_sequence_length)
    sequences_padded = pad_sequences(sequences, maxlen=max_sequence_length)
    return sequences_padded, tokenizer.word_index


def prepare_datasets(test_size: float = 0.2, val_size: float = 0.1):
    """
    Load, preprocess, and split the dataset into training, validation, and test sets.

    Returns:
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        X_test, y_test: Test data and labels.
    """
    texts, labels = load_and_preprocess_data()
    sequences_padded, word_index = tokenize_and_pad(texts)

    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequences_padded, labels, test_size=test_size, random_state=42, stratify=labels
    )
    # Split train+val into train and validation sets
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=42,
        stratify=y_train_val,
    )

    logging.info(
        "Data split into training (%d samples), validation (%d samples), and test (%d samples).",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), word_index


def main():
    # Prepare the datasets
    (X_train, y_train), (X_val, y_val), (X_test, y_test), word_index = (
        prepare_datasets()
    )

    logging.info("Shape of training data: %s", X_train.shape)
    logging.info("Shape of validation data: %s", X_val.shape)
    logging.info("Shape of test data: %s", X_test.shape)
    logging.info("Vocabulary size: %d", len(word_index))


if __name__ == "__main__":
    main()
