#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from imutils import paths
from tqdm import tqdm
import pandas as pd 
import numpy as np
import shutil
import cv2
import os


TRAIN_DATASET_PATH = "ucfTrainTestlist/trainlist01.txt"
TEST_DATASET_PATH = "ucfTrainTestlist/testlist01.txt"


# In[ ]:


# Open the .txt file which have names of training videos
f = open(TRAIN_DATASET_PATH, "r")
temp = f.read()
videos = temp.split('\n')

# Create a dataframe having video names
train = pd.DataFrame()
train['video_name'] = videos
train = train[:-1]
train.head()


# In[ ]:


# Open the .txt file which have names of test videos
with open(TEST_DATASET_PATH, "r") as f:
    temp = f.read()
videos = temp.split("\n")

# Create a dataframe having video names
test = pd.DataFrame()
test["video_name"] = videos
test = test[:-1]
test.head()


# In[ ]:


def extract_tag(video_path):
    return video_path.split("/")[0]

def separate_video_name(video_name):
    return video_name.split("/")[1]

def rectify_video_name(video_name):
    return video_name.split(" ")[0]

def move_videos(df, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in tqdm(range(df.shape[0])):
        videoFile = df['video_name'][i].split("/")[-1]
        videoPath = os.path.join("data", videoFile)
        shutil.copy2(videoPath, output_dir)
    print()
    print(f"Total videos: {len(os.listdir(output_dir))}")


# In[ ]:


train["tag"] = train["video_name"].apply(extract_tag)
train["video_name"] = train["video_name"].apply(separate_video_name)
train.head()


# In[ ]:


train["video_name"] = train["video_name"].apply(rectify_video_name)
train.head()


# In[ ]:


test["tag"] = test["video_name"].apply(extract_tag)
test["video_name"] = test["video_name"].apply(separate_video_name)
test.head()


# In[ ]:


n = 10
topNActs = train["tag"].value_counts().nlargest(n).reset_index()["index"].tolist()
train_new = train[train["tag"].isin(topNActs)]
test_new = test[test["tag"].isin(topNActs)]
train_new.shape, test_new.shape


# In[ ]:


train_new = train_new.reset_index(drop=True)
test_new = test_new.reset_index(drop=True)


# In[ ]:


train_new.to_csv("train.csv", index=False)
test_new.to_csv("test.csv", index=False)


# In[ ]:




