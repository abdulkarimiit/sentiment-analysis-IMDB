# 🎬 IMDB Movie Review Sentiment Analysis

An end-to-end Natural Language Processing (NLP) project that classifies IMDB movie reviews as either **Positive** or **Negative**. This project explores traditional machine learning approaches (**Logistic Regression**, **XGBoost**) as well as a Deep Learning approach using a **Bidirectional LSTM** neural network in TensorFlow/Keras.

---

## 📊 Dataset
* **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (via `kagglehub`)
* **Size:** 50,000 movie reviews evenly split into positive and negative classes.

---

## 🛠️ Data Preprocessing & Cleaning
To prepare raw text for modeling, the pipeline performs the following steps:
1. **HTML Tag Removal:** Strips out HTML formatting (`<br />`, etc.).
2. **Noise Reduction:** Removes special characters, numbers, and punctuation using regular expressions.
3. **Normalization:** Converts all characters to lowercase.
4. **Stemming:** Applies NLTK's `PorterStemmer` to reduce words to their root forms.
5. **Stopword Removal:** Filters out common English stopwords (e.g., "the", "is", "at").
6. **Deduplication:** Removes duplicate rows to maintain data integrity.

---

## 🤖 Models & Performance

| Model | Representation | Test Accuracy |
| :--- | :--- | :--- |
| **Logistic Regression** | TF-IDF (5,000 max features) | ~88.01% |
| **XGBoost** | TF-IDF (5,000 max features) | ~85.05% |
| **Bidirectional LSTM** | Tokenizer + Embedding + Padding | **~88.12%** |

### Bidirectional LSTM Architecture:
* **Embedding Layer:** Maps word indices to dense vectors ($128$-dimensional output).
* **Bidirectional LSTM Layer:** Captures contextual dependencies from both past and future directions ($32$ units).
* **Dropout Layers:** Used for regularization ($0.3$ and $0.5$) to prevent overfitting.
* **Dense Layer:** $64$ units with ReLU activation and L2 regularization.
* **Output Layer:** Single neuron with Sigmoid activation for binary classification.
* **Callbacks:** Includes `EarlyStopping` (patience = 1) and `ReduceLROnPlateau` for optimal training dynamics.

---

## 🚀 Getting Started & Code Usage

### Prerequisites
Make sure you have the required libraries installed:
```bash
pip install pandas numpy nltk scikit-learn xgboost tensorflow matplotlib kagglehub
