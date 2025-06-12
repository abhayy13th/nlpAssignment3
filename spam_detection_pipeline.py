
#!/usr/bin/env python3
"""Spam Detection Pipeline

This script:
1. Loads the 'spam_and_ham_dataset.csv' from the current directory.
2. Cleans the text (removes URLs, non-alphanumeric characters, lowercases).
3. Splits into train/validation/test sets (80/10/10 stratified).
4. Extracts TF-IDF features (unigrams + bigrams, ignore <3 or >85%).
5. Trains Logistic Regression and Linear SVM baselines.
6. Evaluates and prints metrics (accuracy, precision, recall, F1, confusion matrix).
7. Saves the best model and the TF-IDF vectorizer.
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import os

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # 1. Load data
    csv_path = 'spam_and_ham_dataset.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Preprocess
    df['clean_text'] = df['text'].apply(clean_text)

    # 3. Split data
    train_df, temp_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

    # 4. Feature extraction
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.85, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_val   = vectorizer.transform(val_df['clean_text'])
    X_test  = vectorizer.transform(test_df['clean_text'])

    # Encode labels: ham=0, spam=1
    label_map = {'ham': 0, 'spam': 1}
    y_train = train_df['label'].map(label_map).values
    y_val   = val_df['label'].map(label_map).values
    y_test  = test_df['label'].map(label_map).values

    # 5. Train baselines
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    print("Training Linear SVM...")
    svm = LinearSVC(random_state=42)
    svm.fit(X_train, y_train)

    # 6. Evaluate on validation set
    print("\nValidation Results:")
    for name, model in [('Logistic Regression', lr), ('Linear SVM', svm)]:
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', pos_label=1)
        cm = confusion_matrix(y_val, y_pred)
        print(f"\n-- {name} --")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

    # 7. Choose best model based on F1; assume SVM if higher
    # Here simply save both for later comparison
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr, 'models/logistic_model.joblib')
    joblib.dump(svm, 'models/svm_model.joblib')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    print("\nModels and vectorizer saved in 'models/' directory.")

if __name__ == '__main__':
    main()
