# Twitter Sentiment Analysis Model

## Project Overview
This project involves building a sentiment analysis model for Twitter data using Python. The model classifies tweets as either positive or negative. The main steps include data extraction, preprocessing, vectorization, model training, evaluation, and saving the trained model for future use. This README provides a detailed explanation of the code, the technologies used, and instructions for setting up and running the project.

## Table of Contents
1. [Technologies and Skills Used](#technologies-and-skills-used)
2. [Data Extraction](#data-extraction)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Model Saving and Loading](#model-saving-and-loading)
6. [Sentiment Checker Function](#sentiment-checker-function)
7. [Resources](#resources)

## Technologies and Skills Used
- **Python**: Programming language used for the entire project.
- **Kaggle API**: For downloading the dataset.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **NLTK**: For natural language processing, including tokenization and stemming.
- **Scikit-learn**: For machine learning tasks, including data splitting, vectorization, model training, and evaluation.
- **Logistic Regression**: The machine learning algorithm used for sentiment classification.
- **Pickle**: For saving and loading the trained model.

## Data Extraction
1. **Setting up Kaggle Credentials**: To access the Kaggle API, set up the Kaggle credentials and configure the path of `kaggle.json`.
2. **Extracting the CSV Dataset**: Extract the CSV file from the downloaded zip file.

## Data Preprocessing
1. **Loading Data**: Load the data into a Pandas DataFrame.
2. **Label Adjustment**: Replace the label 4 with 1 to standardize the target labels.
3. **Text Cleaning and Stemming**: Clean, tokenize, remove stopwords, and stem the words to standardize the text.

## Model Training and Evaluation
1. **Data Splitting**: Split the data into training and testing sets.
2. **Vectorization**: Convert text data into numerical vectors using TfidfVectorizer.
3. **Model Training**: Train the logistic regression model.
4. **Model Evaluation**: Evaluate the model's accuracy on training and testing data.

## Model Saving and Loading
1. **Saving the Model**: Save the trained model using Pickle.
2. **Loading the Model**: Load the saved model for future use.

## Sentiment Checker Function
A function to check the sentiment of a given tweet using the trained model.

## Resources
- [YouTube Video](https://youtu.be/4YGkfAd2iXM?si=SDC_RAdtGnnEo0Ru): The tutorial followed to create this project.
