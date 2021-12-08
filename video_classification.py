#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import os
import csv
import cv2
from collections import deque
import sys

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

from packaging import version
from matplotlib import pyplot as plt

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."

""" Set Hyper parameters """
MAX_SEQ_LENGTH = 100
NUM_FEATURES = 2048
IMG_SIZE = 128
NUM_EPOCHS = 20
IMG_CHANNELS = 3  ## Change this to 1 for grayscale.
COLOUR_MODE = "rgb"
BATCH_SIZE = 32

# set dir of files
TRAIN_DATASET_PATH = "train.csv"
TEST_DATASET_PATH = "test.csv"
ROOT_DATASET_PATH = "dataset/UCF-101/"
SAVED_MODEL_PATH = "saved_model/"

AUTOTUNE = tf.data.AUTOTUNE
AUGMENTATION = False
TRAIN_MODE = True
GENERATE_DATASET = True


# In[ ]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

center_crop_layer = tf.keras.layers.CenterCrop(IMG_SIZE, IMG_SIZE)


# In[ ]:


def crop_center(frame):
    cropped = center_crop_layer(frame[None, ...])
    cropped = cropped.numpy().squeeze()
    return cropped

# Following method is modified from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def load_video(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center(frame)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.densenet.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")


# In[ ]:


feature_extractor = build_feature_extractor()


# In[ ]:


# Label preprocessing with StringLookup.
label_processor = tf.keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"]), mask_token=None
)
print(label_processor.get_vocabulary())


# In[ ]:


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    df['video_paths'] = df['tag'] + "/" + df['video_name']
    video_paths = df["video_paths"].values.tolist()
    # video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    # print(labels)
    labels = label_processor(labels[..., None]).numpy()
    # print(video_paths)
    # print(labels)
    # `frame_features` are what we will feed to our sequence model.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


if GENERATE_DATASET:
    train_data, train_labels= prepare_all_videos(train_df, ROOT_DATASET_PATH)
    test_data, test_labels= prepare_all_videos(test_df, ROOT_DATASET_PATH)
    
#     np.save("train_data.npy", train_data)
#     np.save("train_labels.npy", train_labels)
    
#     np.save("test_data.npy", test_data)    
#     np.save("test_labels.npy", test_labels)

# train_data, train_labels = np.load("train_data.npy"), np.load("train_labels.npy")
# test_data, test_labels = np.load("test_data.npy"), np.load("test_labels.npy")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")


# In[ ]:


def build_our_model(nb_classes):
    
    frame_features_input = tf.keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = tf.keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.LSTM(200, return_sequences=True)(
        frame_features_input, mask=mask_input
    )

    x = keras.layers.LSTM(200, return_sequences=True)(x)

    x = keras.layers.GRU(20)(x)
    #x = keras.layers.Dropout(0.4)(x)


    x = keras.layers.Dense(2048, activation="relu")(x)
    x = keras.layers.Dense(1024, activation="relu")(x)

    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)

    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    
    output = tf.keras.layers.Dense(nb_classes, activation="softmax")(x)

    rnn_model = tf.keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


# In[ ]:


# Utility for running experiments.
def run_experiment():
    class_vocab = label_processor.get_vocabulary()
    seq_model = build_our_model(len(class_vocab))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        SAVED_MODEL_PATH, save_weights_only=True, save_best_only=True, verbose=1
    )
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.2,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint],
    )
    seq_model.load_weights(SAVED_MODEL_PATH)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model

_, sequence_model = run_experiment()

