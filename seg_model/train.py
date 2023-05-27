"""Function to fit model to training data"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from model import unet_model
from metrics import BinaryMeanIoU
import logging, os

logging.getLogger('monitor').setLevel(logging.INFO)

def train(data={}):
    """
    Trains the semantic segmentation model

    Inputs: 
    data - dictionary containing values including: image_path, mask_path, seed, image_size, batch_size, num_filters,
           epochs, steps_per_epoch, validation_split
    """
    IMAGE_PATH = data.get('image_path', 'dataset/IMG')
    MASK_PATH = data.get('image_path', 'dataset/MSK')
    IMG_HEIGHT = data.get('img_height', 256)
    IMG_WIDTH = data.get('img_width', 256)
    NUM_CHANNELS = data.get('num_channels', 1)
    IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

    NUM_FILTERS = data.get('num_filters', 32)

    NO_OF_FILES = data.get('num_files', len(os.listdir('dataset/IMG/images')))
    TRAIN_LENGTH = NO_OF_FILES - (NO_OF_FILES * 0.2) # 0.2 is validation split
    VAL_LENGTH = NO_OF_FILES * 0.2
    BATCH_SIZE = data.get('batch_size', 32)
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    EPOCHS = data.get('num_epochs', 200)
    VALIDATION_SPLIT = data.get('validation_split', 0.2)
    VALIDATION_STEPS = VAL_LENGTH//BATCH_SIZE
    SEED = data.get('seed', 42)
    # Data generator for loading the image files in batches
    np.random.seed(SEED) # to get the same sampling each time this is run
    tf.random.set_seed(SEED)
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=VALIDATION_SPLIT)

    """Prepare Data to be fed to the model"""
    train_image_generator = train_datagen.flow_from_directory(
        IMAGE_PATH, target_size=IMAGE_SIZE, class_mode=None, batch_size=BATCH_SIZE,
        color_mode='grayscale',subset='training',seed=SEED
    )

    train_mask_generator = train_datagen.flow_from_directory(
        MASK_PATH, target_size=IMAGE_SIZE, class_mode=None, batch_size=BATCH_SIZE,
        color_mode='grayscale', subset='training', seed=SEED
    )

    val_image_generator = train_datagen.flow_from_directory(
        IMAGE_PATH, target_size=IMAGE_SIZE, class_mode=None, batch_size=BATCH_SIZE,
        color_mode='grayscale', subset='validation', seed=SEED
    )

    val_mask_generator = train_datagen.flow_from_directory(
        MASK_PATH, target_size=IMAGE_SIZE, class_mode=None, batch_size=BATCH_SIZE,
        color_mode='grayscale', subset='validation', seed=SEED
    )

    train_generator = zip(train_image_generator, train_mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)


    """Model Section"""
    model = unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), n_filters=NUM_FILTERS)
    logging.info(f"Model summary: \n{model.summary()}")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', BinaryMeanIoU()])

    checkpoint_cb = ModelCheckpoint("best_model.h5", monitor="val_mean_iou", mode="max",verbose=1, save_weights_only=False)
    earlystopping = EarlyStopping(patience=10, verbose=1, monitor='val_mean_iou', mode='max', restore_best_weights=True)

    model_history = model.fit(
        train_generator, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=val_generator,
        callbacks=[checkpoint_cb, earlystopping]
    )

    return model_history
