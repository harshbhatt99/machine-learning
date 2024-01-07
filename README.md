# Machine Learning Projects
This repository contains the codes of different mini projects on ML. It contains the codes and notebooks for the projects that uses conventional machine learning algorithms like  • Linear Regression  • Logistic Regression  • Support Vector Machines (SVM)  • K-Nearest Neighbours (KNN)  • Decision Tree Classifier  • Random Forest Classifier  • KMeans Clustering. 

It also contains the codes on how to use various feature extraction methods like  • Principle Component Analysis (PCA)  • Pearson Correlation (cor_selector)  • Chi-Squared  • Recursive Feature Elimination (RFE)  • Embedded Logistic Regression  • Embedded Random Forest  • Embedded LGBM (Light Gradient Boosting Model)

Here are the details of each mini projects.

### Auto Feature Selector
This file takes the dataset as input and provides best features for the given dataset. It performs the pre-processing (cleaning) function and then runs through the function containing various feature selection methods and returns the best features

### Digit Clustering KMeans
This mini project uses KMeans clustering algorithm to perform Unsupervised Learning on the digits 0-9. It creates the clusters and then predicts the digit (cluster) for the test data.

### Eigen Faces PCA
This mini project demonstrates the dimensionality reduction using PCA. It takes the images of famous faces and reduces the dimensions of the images without loosing the details. This code can be used to append any complex image processing tasks in order to reduce the computational usage.

### ML Algorithms Comparison
This file compares different ML algorithms for a sample tabular dataset. It measures the performace of the algorithms based on the accuracy. This code compares the algorithms like Decision Tree Classifier, KMeans, Logistic Regression, SVM, Random Forest Classifier, and KNN.

### Random Forest Algorithm (BRFSS Dataset)
This notebook takes a sample dataset of BRFSS (Behavioral Risk Factor Survelliance System) and performs the machine learning to predict the class of the health based on given features. It performs the pre-processing on the data and then uses RandomForestCV to test the performance of the algorithm with different hyper parameters. It plots the ROC curve and confusion matrix as a part of the model evaluation. At the end, it also displays the decision tree of the optimized forest to understand the process of the algorithm.
