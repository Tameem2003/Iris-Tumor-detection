import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Constants
IMG_SIZE = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE = 16
EPOCHS = 100

def setup_dataset():
    """Create train/test split from the original dataset structure"""
    try:
        # Create temporary directories for training and testing
        base_dir = "dataset_split"
        for split in ['train', 'test']:
            for category in ['healthy', 'tumorous']:
                os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)

        # Print current working directory and check if folders exist
        print(f"Current working directory: {os.getcwd()}")
        affected_path = "Iris_Tumor_Detection-main/Affected eyes"
        normal_path = "Iris_Tumor_Detection-main/normal eyes"
        
        print(f"Checking if directories exist:")
        print(f"Affected eyes path exists: {os.path.exists(affected_path)}")
        print(f"Normal eyes path exists: {os.path.exists(normal_path)}")

        if not os.path.exists(affected_path) or not os.path.exists(normal_path):
            raise FileNotFoundError(f"Dataset directories not found. Please ensure the following paths exist:\n{affected_path}\n{normal_path}")

        # Get list of files from original directories
        affected_files = os.listdir(affected_path)
        normal_files = os.listdir(normal_path)

        print(f"Found {len(affected_files)} affected eye images")
        print(f"Found {len(normal_files)} normal eye images")

        # Split files into train and test sets
        affected_train, affected_test = train_test_split(affected_files, test_size=0.2, random_state=42)
        normal_train, normal_test = train_test_split(normal_files, test_size=0.2, random_state=42)

        # Copy files to new structure
        print("Copying affected eyes images...")
        for file in affected_train:
            shutil.copy2(
                os.path.join(affected_path, file),
                os.path.join(base_dir, "train", "tumorous", file)
            )
        for file in affected_test:
            shutil.copy2(
                os.path.join(affected_path, file),
                os.path.join(base_dir, "test", "tumorous", file)
            )

        print("Copying normal eyes images...")
        for file in normal_train:
            shutil.copy2(
                os.path.join(normal_path, file),
                os.path.join(base_dir, "train", "healthy", file)
            )
        for file in normal_test:
            shutil.copy2(
                os.path.join(normal_path, file),
                os.path.join(base_dir, "test", "healthy", file)
            )

        print("Dataset setup completed successfully!")
        return base_dir

    except Exception as e:
        print(f"\nError during dataset setup: {str(e)}")
        print("\nPlease ensure your dataset is organized as follows:")
        print("Iris_Tumor_Detection-main/")
        print("├── Affected eyes/")
        print("│   └── [affected eye images]")
        print("└── normal eyes/")
        print("    └── [normal eye images]")
        raise

def create_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    
    return model, base_model

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    return train_datagen, test_datagen

def train_model():
    print("Setting up dataset...")
    base_dir = setup_dataset()

    print("Creating data generators...")
    train_datagen, test_datagen = create_data_generators()

    # Calculate class weights
    total_healthy = len(os.listdir(os.path.join(base_dir, "train", "healthy")))
    total_tumorous = len(os.listdir(os.path.join(base_dir, "train", "tumorous")))
    total = total_healthy + total_tumorous
    
    class_weights = {
        0: (total / (2 * total_healthy)),  # healthy
        1: (total / (2 * total_tumorous))  # tumorous
    }

    print("Loading and preparing the data...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    print("Creating and compiling model...")
    model, base_model = create_model()

    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=8,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            'iris_tumor_cnn_model.h5',
            monitor='val_loss',
            save_best_weights_only=True,
            mode='min'
        )
    ]

    print("Training model (Phase 1 - Training only top layers)...")
    history1 = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print("\nFine-tuning the model...")
    # Unfreeze the base model gradually
    base_model.trainable = True
    
    # Freeze first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )

    print("Training model (Phase 2 - Fine-tuning)...")
    history2 = model.fit(
        train_generator,
        epochs=40,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print("\nEvaluating model...")
    test_results = model.evaluate(test_generator, verbose=1)
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    
    print("\nTest Results:")
    for metric, value in zip(metrics, test_results):
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Calculate F1 Score
    precision = test_results[2]  # index 2 is precision
    recall = test_results[3]    # index 3 is recall
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"F1 Score: {f1_score:.4f}")

    # Clean up temporary directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    train_model()
