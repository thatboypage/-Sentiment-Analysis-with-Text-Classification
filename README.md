# 💬 Sentiment Analysis with Text Classification

This project involves performing sentiment analysis on textual data using classical machine learning techniques. The primary goal is to classify text into one of four categories: **positive**, **negative**, **neutral**, or **others**, using NLP preprocessing, TF-IDF vectorization, SMOTE resampling, and classification algorithms.

---

## 🧠 Objective

To build a robust and efficient machine learning pipeline that can accurately detect sentiment from raw text data and evaluate various models for performance.

---

## 🛠️ Tools & Libraries

* **Python**
* `pandas`, `numpy`, `matplotlib`, `seaborn`
* `sklearn`, `imblearn`, `nltk`
* `LazyPredict`, `wordcloud`

---

## 📂 Project Structure

```
├── Sentiment_analysis.ipynb
├── README.md
├── Test-Set.csv
└── requirements.txt (optional)
```

---

## 📊 Dataset Overview

The dataset consists of customer feedback labeled with sentiments:

* **Text** – raw feedback.
* **Sentiment** – class label: `positive`, `negative`, `neutral`, `others`.

---

## 🔄 Preprocessing Steps

1. **Lowercasing & Tokenization**
2. **Stopword Removal** (using NLTK)
3. **TF-IDF Vectorization**
4. **SMOTE Oversampling** to handle class imbalance
5. **Train-Test Split**

---

## ⚙️ Modeling Workflow

### 🧪 Model Benchmarking (LazyPredict)

Ran LazyPredict on the processed data to benchmark 20+ models. Top performers included:

| Model                         | Accuracy | F1 Score | Time Taken |
| ----------------------------- | -------- | -------- | ---------- |
| GaussianNB                    | 1.00     | 1.00     | 0.54 sec   |
| SGDClassifier                 | 1.00     | 1.00     | 1.27 sec   |
| QuadraticDiscriminantAnalysis | 1.00     | 1.00     | 3.01 sec   |
| CalibratedClassifierCV        | 1.00     | 1.00     | 215.22 sec |

---

### ✅ Final Model: Gaussian Naive Bayes

#### 🔍 Hyperparameter Tuning

Performed GridSearchCV to optimize `var_smoothing` parameter.

```python
grid_params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
```

#### 📈 Final Evaluation Metrics

* **Accuracy**: `99.6%`
* **Macro F1-Score**: `1.00`

| Sentiment | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Negative  | 1.00      | 1.00   | 1.00     | 139     |
| Neutral   | 0.99      | 1.00   | 1.00     | 132     |
| Others    | 1.00      | 0.99   | 0.99     | 137     |
| Positive  | 0.99      | 1.00   | 1.00     | 124     |

---

## ☁️ Word Cloud Visualization

A word cloud was generated to identify the most frequent terms used across the text data.

![WordCloud](./wordcloud.png)

---

## 🗣️ Top Words Per Sentiment

Using `collections.Counter`, the top 10 most frequent words were extracted for each sentiment class.

---

## 📌 Key Insights

* **Gaussian Naive Bayes** outperformed other models with high accuracy and F1 scores.
* **TF-IDF + SMOTE** preprocessing greatly improved classification consistency.
* Most frequent sentiment-indicative words align well with human interpretation.
* Efficient, interpretable, and high-performing baseline model for sentiment detection.

---

## 🔗 Connect with Me

Let’s connect if you’re working on:

* Customer feedback or retention
* Applied ML/NLP projects
* Or just want to chat about data science!

📬 [LinkedIn]([https://www.linkedin.com/in/richard-olanite-55b4b0241/])
