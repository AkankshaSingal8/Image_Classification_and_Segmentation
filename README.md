# Image Classification and Segmentation Project

## Overview
This project involves a series of tasks focused on image classification and segmentation using datasets such as the Russian Wildlife Dataset, Indian Driving Dataset (IDD), and Cityscapes Dataset. We trained custom Convolutional Neural Networks (CNN) from scratch, fine-tuned pre-trained models, performed data augmentation, and evaluated the performance of these models on test sets. The project also includes cross-domain analysis between different datasets.

## Tasks

### 1. Image Classification with Russian Wildlife Dataset

#### 1.1 Dataset Preparation
- Downloaded the Russian Wildlife Dataset.
- Mapped the class labels: 
  - 'amur leopard' : 0 
  - 'amur tiger' : 1 
  - 'birds' : 2 
  - 'black bear' : 3 
  - 'brown bear' : 4 
  - 'dog' : 5 
  - 'roe deer' : 6 
  - 'sika deer' : 7 
  - 'wild boar' : 8 
  - 'people' : 9
- Performed a stratified random split into training (70%), validation (10%), and test sets (20%).
- Created a custom Dataset class and initialized Weights & Biases (WandB) for experiment tracking.

#### 1.2 Data Loaders and Visualization
- Created data loaders for the train, validation, and test sets using PyTorch.
- Visualized the data distribution across class labels for training and validation sets.

### 2. Training a CNN from Scratch

#### 2.1 CNN Architecture
- Designed a CNN with 3 convolution layers:
  - 1st layer: 32 filters, 3x3 kernel, stride 1, padding 1, followed by a 4x4 Max Pooling.
  - 2nd layer: 64 filters, 3x3 kernel, stride 1, padding 1, followed by a 2x2 Max Pooling.
  - 3rd layer: 128 filters, 3x3 kernel, stride 1, padding 1, followed by a 2x2 Max Pooling.
- Added a classification head and used ReLU activation functions.

#### 2.2 Model Training and Evaluation
- Trained the model using Cross-Entropy Loss and the Adam optimizer for 10 epochs, logging metrics with WandB.
- Analyzed the training and validation loss plots for signs of overfitting.
- Reported the accuracy, F1-Score, and confusion matrix on the test set.
- For each class, visualized 3 misclassified images, discussed potential causes, and proposed possible workarounds.

### 3. Fine-Tuning a Pre-Trained Model

#### 3.1 Fine-Tuning ResNet-18
- Trained a classifier with a fine-tuned ResNet-18 (pre-trained on ImageNet) following the same procedure as in the CNN.
- Logged loss and accuracy, and analyzed overfitting based on the training and validation loss plots.
- Reported accuracy, F1-Score, and confusion matrix on the test set.
- Visualized feature vectors using t-SNE plots in both 2D and 3D spaces for the training and validation sets.

### 4. Data Augmentation Techniques

#### 4.1 Data Augmentation and Training
- Applied three or more data augmentation techniques to enhance the dataset variety.
- Trained the model using the same procedure and compared the impact on overfitting.
- Reported accuracy, F1-Score, and confusion matrix on the test set.

### 5. Euclidean Distance Analysis
- Analyzed the misclassified samples from the CNN by calculating Euclidean distances in the feature space.
- Compared the feature distances for the same samples across all trained models and discussed the results.

### 6. Model Performance Comparison
- Compared the performance of all three models, including the CNN from scratch, fine-tuned ResNet-18, and the model trained with augmented data.

### 7. Image Segmentation with Indian Driving Dataset

#### 7.1 Dataset Preparation
- Downloaded and created a dataloader for the Indian Driving Dataset.
- Visualized the data distribution and provided examples of images with corresponding masks.

#### 7.2 Evaluating a Segmentation Model
- Used a pre-trained DeepLabv3Plus-Pytorch model for inference on 30% of the IDD.
- Resized images to 512x512 and evaluated classwise performance using metrics like pixel-wise accuracy, dice coefficient, mAP, IoU, precision, and recall.
- Visualized images with IoU ≤ 0.5 and discussed potential failure reasons.

### 8. Analysis of Segmentation Results
- Created and visualized a confusion matrix for the segmentation model.
- Analyzed the precision, recall, and F1 score for each class and identified areas for improvement.

### 9. Cross-Domain Analysis
- Performed inference on the Cityscapes validation set using the DeepLabv3Plus model.
- Evaluated and compared confusion matrices for Cityscapes and IDD datasets, and discussed the reasons for the observed performance differences.

## Conclusion
The project provides insights into image classification and segmentation, including the challenges of working with different datasets and the importance of data augmentation and fine-tuning. The results highlight the performance variations across different models and datasets, offering valuable lessons for improving model generalization and robustness.
