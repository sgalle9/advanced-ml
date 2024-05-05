# Mitral Valve Segmentation

## 1. Task

This project aims to segment the mitral valve in echocardiograms, one of the four heart valves that regulate blood flow. Accurate segmentation serves as a first step in automating the assessment of valve functionality, helping with the detection of related cardiac diseases.


## 2. Data

**Disclaimer:** The data, containing sensitive information, is not publicly available.

The dataset includes echocardiograms of variable lengths, with only three frames per video labeled. It originates from two distinct sources, which differ in video resolution and the labelers' expertise levels (i.e., "amateur" vs. "expert").


## 3. Methodology

### 3.1. Preprocessing

Given the variable lengths and resolutions of the echocardiogram frames, resizing was necessary. Since only three frames per video were labeled, we focused exclusively on these frames for our supervised learning task, disregarding the unlabeled parts of the videos. Due to the small size of the dataset (approximately 50 patients, one video each) we employed random transformations to augment the data. 

N.B. We made sure that the transformations chosen produce realistic variations, typical of what might be seen in clinical settings, so as to prevent the model from overfitting on unlikely scenarios.


### 3.2. Model

The U-Net architecture was chosen for the segmentation task.


### 3.3. Loss function

We modified the typical cross-entropy loss to include two types of weighting. Firstly, data-source-specific weights prioritized higher-quality observations (i.e., those labeled by experts). Secondly, class-specific weights addressed the issue of class imbalance resulting from the mitral valve occupying a small area of the image.


## 4. Evaluation

The Jaccard similarity coefficient is used to assess the similarity between the predicted segmentation and the ground truth.
