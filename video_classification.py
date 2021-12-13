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
import gc
import pickle

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

from packaging import version
from matplotlib import pyplot as plt

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."

""" Set Hyper parameters """
MAX_SEQ_LENGTH = 100
NUM_FEATURES = 2048
IMG_SIZE = 224
NUM_EPOCHS = 20
IMG_CHANNELS = 3  ## Change this to 1 for grayscale.
BATCH_SIZE = 32

# set dir of files
TRAIN_DATASET_VERSION = "trainfull"
TEST_DATASET_VERSION = "testfull"
TRAIN_DATASET_PATH = TRAIN_DATASET_VERSION+".csv"
TEST_DATASET_PATH = TEST_DATASET_VERSION+".csv"
ROOT_DATASET_PATH = "dataset/UCF-101/"
SAVED_MODEL_PATH = "saved_model/"

AUTOTUNE = tf.data.AUTOTUNE
AUGMENTATION = False
TRAIN_MODE = True
GENERATE_DATASET = True
RETRAIN_MODEL = False


# In[ ]:


train_df = pd.read_csv(TRAIN_DATASET_PATH)
test_df = pd.read_csv(TEST_DATASET_PATH)

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")


# In[ ]:


# def crop_center_square(frame):
#     y, x = frame.shape[0:2]
#     min_dim = min(y, x)
#     start_x = (x // 2) - (min_dim // 2)
#     start_y = (y // 2) - (min_dim // 2)
#     return frame[start_y : start_y+min_dim, start_x : start_x+min_dim]

# Following method is modified from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS),
    )
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
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
    df['video_paths'] = root_dir + df['tag'] + "/" + df['video_name']
    video_paths = df["video_paths"].values.tolist()
    # video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    # print(labels)
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx,path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.hike intern
        #path = video_paths[idx]
        frames = load_video(path)
        frames = frames[None, ...]

        gc.collect()

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_featutes = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            try:
                video_length = batch.shape[1]
                length = min(MAX_SEQ_LENGTH, video_length)
                for j in range(length):
                    temp_frame_featutes[i, j, :] = feature_extractor.predict(batch[None, j, :])
                temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
                frame_features[idx,] = temp_frame_featutes.squeeze()
                frame_masks[idx,] = temp_frame_mask.squeeze()
            except:
                #print(i, j, length)
                pass

        gc.collect()
        print(idx)

    return (frame_features, frame_masks), labels

gc.collect()

if GENERATE_DATASET:
    train_data, train_labels= prepare_all_videos(train_df, ROOT_DATASET_PATH)
    test_data, test_labels= prepare_all_videos(test_df, ROOT_DATASET_PATH)

    with open(TRAIN_DATASET_VERSION+'_data.pkl','wb') as f:
        pickle.dump(train_data, f)

    with open(TRAIN_DATASET_VERSION+'_labels.pkl','wb') as f:
        pickle.dump(train_labels, f)

    with open(TEST_DATASET_VERSION+'_data.pkl','wb') as f:
        pickle.dump(test_data, f)

    with open(TEST_DATASET_VERSION+'_labels.pkl','wb') as f:
        pickle.dump(test_labels, f)



f = open(TRAIN_DATASET_VERSION+'_data.pkl', 'rb')
train_data = pickle.load(f)
f.close()

f = open(TRAIN_DATASET_VERSION+'_labels.pkl', 'rb')
train_labels = pickle.load(f)
f.close()

f = open(TEST_DATASET_VERSION+'_data.pkl', 'rb')
test_data = pickle.load(f)
f.close()

f = open(TEST_DATASET_VERSION+'_labels.pkl', 'rb')
test_labels = pickle.load(f)
f.close()


print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")


# In[ ]:


def confusion_matrix_report(labels, predicts, target_names):
    confusion = confusion_matrix(labels, predicts)
    print('Confusion Matrix\n')
    print(confusion)
    
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(labels, predicts)))

    print('Micro Precision: {:.2f}'.format(precision_score(labels, predicts, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(labels, predicts, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(labels, predicts, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(labels, predicts, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(labels, predicts, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(labels, predicts, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(labels, predicts, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(labels, predicts, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(labels, predicts, average='weighted')))

    print('\nClassification Report\n')
    print(classification_report(labels, predicts, target_names=target_names))


# In[ ]:


def plot_epoch_result(epochs, loss, name, model_name, colour):
    plt.plot(epochs, loss, colour, label=name)
#     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(model_name+ '_'+name+'_epoch_result.png')
    plt.show()
    
class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self,
                 model_path,
                 n_model
                ):
        super(CustomSaver, self).__init__()
        self.history = {}
        self.epoch = []
        self.model_path = model_path
    
        self.name_model = n_model
        self.custom_loss = []
        self.epochs_list = []
            
    def on_train_end(self, logs=None):
        print(self.model_path)
        self.model.save(self.model_path)
        
        plot_epoch_result(self.epochs_list, self.custom_loss, "Loss", self.name_model, "g")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epoch.append(epoch)
        for k, v in logs.items():
#             print(k, v)
            self.history.setdefault(k, []).append(v)
        
        self.epochs_list.append(epoch)
        self.custom_loss.append(logs["loss"])

        if (epoch + 1) % 15 == 0:
            self.model.save_weights(self.model_path)
            print('saved for epoch',epoch + 1)


# In[ ]:


def build_our_model(nb_classes):
    
    frame_features_input = tf.keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = tf.keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = tf.keras.layers.LSTM(200, return_sequences=True)(
        frame_features_input, mask=mask_input
    )

    x = tf.keras.layers.LSTM(200, return_sequences=True)(x)

    x = tf.keras.layers.GRU(20)(x)
    #x = keras.layers.Dropout(0.4)(x)


    x = tf.keras.layers.Dense(2048, activation="relu")(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    
    output = tf.keras.layers.Dense(nb_classes, activation="softmax")(x)

    rnn_model = tf.keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


# In[ ]:


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames

def testing_stage(model, df, root_dir):
    print("testing start")
    num_samples = len(df)
    df['video_paths'] = root_dir + df['tag'] + "/" + df['video_name']
    video_paths = df["video_paths"].values.tolist()
    labels = df["tag"].values
    # labels = label_processor(labels[..., None]).numpy().flatten()
    class_vocab = label_processor.get_vocabulary()
    predictions = []
    name_list = []
    # print(labels)
    for idx, path in enumerate(video_paths):
        print(path)
        frames = load_video(path)
        frame_features, frame_mask = prepare_single_video(frames)
        probabilities = sequence_model.predict([frame_features, frame_mask])[0]
        probs = np.argsort(probabilities)[::-1]
        name_image = os.path.basename(path)
        print(name_image)
        predictions.append(class_vocab[probs[0]])
        name_list.append(name_image)
        # print(class_vocab[probs[0]], probs[0])
        for i in probs:
            print(f"{class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    
    
    confusion_matrix_report(labels, predictions, class_vocab)
    
    
    print("created csv for the result.")
    with open('predictions_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageName', 'Label'])
        writer.writerows(zip(name_list, predictions))


# In[ ]:


# Utility for running experiments.
def run_experiment():
    name_model = str(IMG_SIZE)+"_UCF101_"+str(NUM_EPOCHS)
    class_vocab = label_processor.get_vocabulary()
    
    seq_model = build_our_model(len(class_vocab))
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        SAVED_MODEL_PATH, save_weights_only=True, save_best_only=True, verbose=1
    )
    
    path_model = SAVED_MODEL_PATH + name_model + "_model" + ".h5"
    print(path_model)
    saver_callback = CustomSaver(
            path_model,
            name_model
        )
    
    if RETRAIN_MODEL:
        print("Model Load Weights.")
        # seq_model.load_weights(path_model)
        seq_model = tf.keras.models.load_model(path_model)
    
    if TRAIN_MODE: 
        history = seq_model.fit(
            [train_data[0], train_data[1]],
            train_labels,
            # validation_split=0.2,
            epochs=NUM_EPOCHS,
            callbacks=[checkpoint, saver_callback],
        )
    # seq_model.load_weights(SAVED_MODEL_PATH)
    
    seq_model = tf.keras.models.load_model(path_model)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return seq_model


# In[ ]:


if __name__ == "__main__":
    print("run experiments")
    sequence_model = run_experiment()
    testing_stage(sequence_model, test_df, ROOT_DATASET_PATH)

