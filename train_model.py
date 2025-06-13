import os
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = 'data'
RAW_DATA_DIR = 'raw_data'
AUGMENTATION_FACTOR = 200  # Images per original
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
TEST_SIZE = 0.15

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# ============================================
# ENHANCED DATA AUGMENTATION
# ============================================
def create_augmentor():
    return ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.2, 1.8],
        fill_mode='nearest',
        channel_shift_range=75.0,
        validation_split=TEST_SIZE
    )

# ============================================
# AUTOMATED DATA GENERATION
# ============================================
def generate_augmented_dataset():
    # Clean existing data
    for dirpath in [BASE_DIR]:
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
    
    # Create directory structure
    os.makedirs(os.path.join(BASE_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'validation'), exist_ok=True)
    
    # Initialize augmentor
    datagen = create_augmentor()
    
    # Process each class
    for class_name in os.listdir(RAW_DATA_DIR):
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        # Create class directories
        os.makedirs(os.path.join(BASE_DIR, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, 'validation', class_name), exist_ok=True)
        
        # Get original images
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split into train/validation
        train_imgs, val_imgs = train_test_split(images, test_size=TEST_SIZE, random_state=42)
        
        # Process validation images
        for img in val_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(BASE_DIR, 'validation', class_name, img)
            shutil.copy(src, dst)
        
        # Generate augmented training images
        for img in train_imgs:
            img_path = os.path.join(class_path, img)
            image = Image.open(img_path).convert('RGB')
            x = img_to_array(image)
            x = x.reshape((1,) + x.shape)
            
            # Generate augmented images
            save_prefix = f"aug_{os.path.splitext(img)[0]}"
            counter = 0
            
            for batch in datagen.flow(x, batch_size=1,
                                    save_prefix=save_prefix,
                                    save_format='jpeg',
                                    save_to_dir=os.path.join(BASE_DIR, 'train', class_name)):
                counter += 1
                if counter >= AUGMENTATION_FACTOR:
                    break

    print(f"Generated {AUGMENTATION_FACTOR}x augmented dataset")

# ============================================
# ENHANCED MODEL TRAINING
# ============================================
def train_model():
    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Model architecture
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Unfreeze last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    outputs = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(base_model.input, outputs)
    
    # Optimizer with mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=4),
        tf.keras.callbacks.ModelCheckpoint('trained_model/best_model.h5', save_best_only=True)
    ]
    
    # Start training
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=os.cpu_count()
    )
    
    # Save final model
    model.save('trained_model/object_detector.h5')
    print("Training complete. Model saved.")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    generate_augmented_dataset()
    train_model()
