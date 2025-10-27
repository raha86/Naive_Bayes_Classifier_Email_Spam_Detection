# Naive Bayes Classifier – Email Spam Detection

A machine learning project to classify emails as Spam or Not Spam (Ham) using the Multinomial Naive Bayes algorithm.
This project demonstrates text preprocessing, feature extraction, and model building for email classification.

## Dataset:

  Columns:
  text → The actual content of the email message.<br>
  spam → Label column:
  1 → Spam
  0 → Not Spam (Ham)

## Problem Statement:

  Build a machine learning model that classifies emails as spam or non-spam based on their content.

## Steps Followed:<br>

### 1. Data Exploration: 
  * Loaded and inspected the dataset (5728 rows, 2 columns)
  * Removed duplicate records (33 duplicates found)
  * Checked for null values (none found)
  * Observed class imbalance in target variable (spam)

### 2. Data Preprocessing:<br>
  * Split dataset into training and testing sets (80:20 ratio) using train_test_split with stratify for balanced representation

### 3. Feature Engineering:<br>

  Used CountVectorizer to convert text data into a numerical representation (Bag-of-Words model):<br>

  from sklearn.feature_extraction.text import CountVectorizer<br>
  cv = CountVectorizer()<br>
  X_train_vectorized = cv.fit_transform(X_train)<br>
  X_test_vectorized = cv.transform(X_test)<br>

  Resulting feature matrix:<br>
  (4556, 33700) → 4556 emails, 33700 unique tokens.<br>

### 4. Model Building:<br>

  Used Multinomial Naive Bayes, suitable for text classification with discrete features:<br>

  from sklearn.naive_bayes import MultinomialNB<br>
  nb_model = MultinomialNB()<br>
  nb_model.fit(X_train_vectorized, y_train)<br>

### 5. Model Evaluation:<br>

  Predicted and evaluated the model using classification_report and ROC-AUC score from sklearn.metrics import classification_report, roc_auc_score<br>
  
  Results:<br>
  
  Metric	Value<br>
  Accuracy	99%<br>
  Precision (Spam)	0.97<br>
  Recall (Spam)	1.00<br>
  F1-Score (Spam)	0.99<br>
  ROC-AUC Score	0.994<br>
