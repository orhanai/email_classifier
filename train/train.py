from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from email_classifier.pre_processing.main import prepare_datasets
from email_classifier.train.main import model


def train():
    (X_train, y_train), (X_val, y_val), (X_test, y_test), word_index = (
        prepare_datasets()
    )
    # Define callbacks
    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        "best_model.h5", monitor="val_loss", save_best_only=True
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,  # Adjust based on your hardware and data
        epochs=10,  # You may need more epochs depending on convergence
        validation_data=(X_val, y_val),
        callbacks=[early_stop, model_checkpoint],
    )

    # Optionally, evaluate on test data after training:
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
