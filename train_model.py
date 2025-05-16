import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# ============================================
# STEP 1: AUTOMATIC DATA ORGANIZATION
# ============================================

def organize_data():
    # Configuration
    RAW_DATA_DIR = 'raw_data'
    BASE_DIR = 'data'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    VAL_DIR = os.path.join(BASE_DIR, 'validation')
    TEST_SIZE = 0.15  # 15% for validation
    
    # Create directories
    for dir_path in [TRAIN_DIR, VAL_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # Process each class
    for class_name in os.listdir(RAW_DATA_DIR):
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Get all images
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split to train/validation
        train_imgs, val_imgs = train_test_split(images, test_size=TEST_SIZE, random_state=42)

        # Copy files
        for img in train_imgs:
            src = os.path.join(class_dir, img)
            dest_dir = os.path.join(TRAIN_DIR, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src, dest_dir)

        for img in val_imgs:
            src = os.path.join(class_dir, img)
            dest_dir = os.path.join(VAL_DIR, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src, dest_dir)

    print("Data organized successfully!")

# ============================================
# STEP 2: MODEL TRAINING
# ============================================

def train_model():
    # Configuration
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 30

    # Data generators
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.7,1.3],
        horizontal_flip=True,
        rescale=1./255
        zoom_range=0.2
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        'data/validation',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Model setup
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # Freeze layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.2)(x)
    outputs = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(base_model.input, outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # Save model
    model.save('trained_model/object_detector.h5')
    print("Model trained and saved!")

# ============================================
# RUN THE FULL PIPELINE
# ============================================

if __name__ == "__main__":
    organize_data()
    train_model()