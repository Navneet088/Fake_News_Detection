# Fake News Detection

## Table of Contents
- [Introduction](#introduction)
- [Objective](#objective)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction
Fake news has become a critical issue in the digital age, spreading misinformation across social media and news platforms.  
This project aims to detect whether a news article is **real** or **fake** using machine learning techniques.

---

## Objective
- Build a machine learning model to classify news articles as **real** or **fake**.  
- Evaluate different models to find the most accurate approach.  
- Preprocess and vectorize text data for effective classification.

---

## Dataset
- **Source:** Kaggle / Custom dataset  
- **Files:**
  - `fake_or_real_news.csv` → contains labeled news articles
  - `Fake.csv` → fake news samples
  - `True.csv` → real news samples
- **Columns:**
  - `title` → News title
  - `text` → News content
  - `label` → `FAKE` or `REAL`

---

## Preprocessing
Text preprocessing is performed to clean the data:

1. Remove punctuation, special characters, and numbers
2. Convert text to lowercase
3. Remove stopwords
4. Lemmatization
5. Combine title and content for training

## Preprocessing
Text preprocessing is performed to clean the data:

1. Remove punctuation, special characters, and numbers
2. Convert text to lowercase
3. Remove stopwords
4. Lemmatization
5. Combine title and content for training

```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)




