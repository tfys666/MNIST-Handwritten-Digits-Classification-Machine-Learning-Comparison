# MNIST-Handwritten-Digits-Classification---Machine-Learning-Comparison

A comprehensive comparison of three machine learning approaches on the classic MNIST handwritten digits dataset, implemented in MATLAB.

## 📊 Project Overview

This project systematically evaluates and compares the performance of different machine learning techniques on the MNIST dataset:

- **K-means Clustering** (Unsupervised Learning)
- **t-SNE Visualization** (Dimensionality Reduction)  
- **SVM Classification** (Supervised Learning)

## 🚀 Prerequisites
- MATLAB R2020b or later
- Statistics and Machine Learning Toolbox

## 🎯 Key Results

| Method | Accuracy | Best Configuration | Key Insight |
|--------|----------|-------------------|-------------|
| K-means Clustering | 59.07% | K=10 | Limited by pixel-space similarity |
| t-SNE Visualization | - | Perplexity=30 | Reveals inherent data structure |
| SVM Classification | **95.93%** | RBF Kernel | Optimal for this task |
