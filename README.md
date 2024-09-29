# Machine Learning Internship Tasks

This repository contains the implementation of various machine learning tasks completed during the Machine Learning Internship at TechWithWarriors. Each task is implemented using different machine learning techniques and datasets.

## Task 01: Linear Regression Model for House Price Prediction

**Description:**
This task involves implementing a linear regression model to predict house prices using the California Housing Dataset. The model is built using scikit-learn, a powerful Python library for machine learning.

**Dataset:**
- **California Housing Dataset**: This dataset contains information on various housing features like the number of rooms, population, and median house values for California districts.

**Implementation Details:**
- **Model**: Linear Regression
- **Features**: Various features from the dataset were used to predict the target variable (house price).
- **Evaluation Metrics**: Mean Squared Error (MSE), R-squared.

**How to Run:**
1. Clone the repository and navigate to the `MachineLearning` directory.
2. Open the `Task 01 :Implement a linear regression model using scikit- learn to predict house prices.ipynb` file in Google Colab or Jupyter Notebook.
3. Run the cells in the notebook to train the model and evaluate its performance.

**Results:**
- The model achieved an R-squared score of **0.66** on the test set, indicating the percentage of variance explained by the model.

---

## Task 02: Decision Tree Classifier for Iris Flower Classification

**Description:**
This task involves building a decision tree classifier to classify iris flowers into three species based on their features. The decision tree algorithm is implemented using scikit-learn.

**Dataset:**
- **Iris Dataset**: This classic dataset contains 150 samples of iris flowers, classified into three species: Setosa, Versicolor, and Virginica.

**Implementation Details:**
- **Model**: Decision Tree Classifier
- **Features**: Sepal length, sepal width, petal length, and petal width.
- **Evaluation Metrics**: Accuracy, Confusion Matrix.

**How to Run:**
1. Clone the repository and navigate to the `MachineLearning` directory.
2. Open the `Task 02:Use a decision tree classifier to classify a dataset of iris flowers..ipynb` file in Google Colab or Jupyter Notebook.
3. Run the cells in the notebook to train the model and evaluate its performance.

**Results:**
- **Precision per class**: [1.0, 0.9375, 0.9091]
- **Recall per class**: [1.0, 0.8824, 0.9524]
- **F1 Score per class**: [1.0, 0.9091, 0.9302]
- The model achieved a balanced performance across the three iris species, indicating good classification accuracy for each class.

---

## Task 03: K-Nearest Neighbors (KNN) Classifier for MNIST Image Classification

**Description:**
This task involves implementing a K-Nearest Neighbors (KNN) classifier to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.

**Dataset:**
- **MNIST Dataset**: This dataset contains 70,000 images of handwritten digits, each image is 28x28 pixels.

**Implementation Details:**
- **Model**: K-Nearest Neighbors (KNN)
- **Libraries Used**: TensorFlow, Keras
- **Evaluation Metrics**: Accuracy, Confusion Matrix.

**How to Run:**
1. Clone the repository and navigate to the `MachineLearning` directory.
2. Open the `Task 03: Develop a k-nearest neighbors (KNN) classifier for image classification using TensorFlow/Keras..ipynb` file in Google Colab or Jupyter Notebook.
3. Run the cells in the notebook to train the model and evaluate its performance.

**Results:**
- The KNN model achieved an accuracy of **97.05%** on the test set, correctly identifying the handwritten digits.

---

## Task 04: Collaborative Filtering-based Recommendation System

**Description:**
This task involves creating a basic recommendation system using collaborative filtering. The system suggests items to users based on their past interactions and preferences.

**Dataset:**
- **MovieLens Dataset**: This dataset contains millions of movie ratings from users, which can be used to build and evaluate recommendation systems.

**Implementation Details:**
- **Method**: Collaborative Filtering
- **Libraries Used**: Surprise , pandas, numpy
- **Evaluation Metrics**: Root Mean Squared Error (RMSE), Precision, Recall.

**How to Run:**
1. Clone the repository and navigate to the `MachineLearning` directory.
2. Open the `Task 04: Create a basic recommendation system using collaborative filtering..ipynb` file in Google Colab or Jupyter Notebook.
3. Run the cells in the notebook to train the model and evaluate its performance.

**Results:**
- The recommendation system achieved an RMSE of 0.8431 on the test set, demonstrating its effectiveness in suggesting relevant items to users.

---

## General Instructions
To view and execute any of the tasks:
1. Clone this repository using:
   ```bash
   git clone https://github.com/Fatemehkiasaveh/TechWithWarriors.git
