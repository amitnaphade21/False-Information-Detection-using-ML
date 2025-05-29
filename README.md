# 📰 False Information Detection using Machine Learning 

This project focuses on detecting fake news articles using Natural Language Processing and supervised Machine Learning models. It uses text features from author, title, and content to classify whether a news piece is real or fake with real time checking by taking input from user provided a userfriendly ui using streamlit library.

Sample Outputs:

![image](https://github.com/user-attachments/assets/a2193472-0f61-4cda-8d28-d81445511853)


---

## 🔍 Project Features

- Preprocessing with stemming and stopword removal
- TF-IDF vectorization for feature extraction
- Model training with:
  - Logistic Regression
  - Random Forest
  - SVM
  - KNN
- Model evaluation (accuracy, precision, recall, F1-score)
- Confusion matrix and accuracy bar plots
- Save model & vectorizer using `joblib` / `pickle`

---

## 📂 Dataset

The project uses a CSV file (`train.csv`) containing the following columns:
- `author`
- `title`
- `text`
- `label` → 0 (Real), 1 (Fake)

---

## 🧹 Data Preprocessing

- Merged `author`, `title`, and `text` columns
- Removed non-alphabetic characters
- Converted text to lowercase
- Removed stopwords
- Applied Porter Stemmer
- Vectorized text using **TF-IDF**

---

## 🤖 Models Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Multinomial Naive Bayes / K-Nearest Neighbors (KNN)

Each model is trained and evaluated on test data split using stratified sampling.

---

## 📊 Model Evaluation Metrics

Each model is evaluated using the following:

- ✅ Accuracy
- 🎯 Precision
- 📈 Recall
- 📊 F1-Score
- 📉 Confusion Matrix

---

## 📈 Visualizations

### 🔹 Model Accuracy Comparison

# 📰 Fake News Detection using Machine Learning and NLP

This project focuses on detecting fake news articles using Natural Language Processing and supervised Machine Learning models. It uses text features from author, title, and content to classify whether a news piece is real or fake.

---

## 🔍 Project Features

- Preprocessing with stemming and stopword removal
- TF-IDF vectorization for feature extraction
- Model training with:
  - Logistic Regression
  - Random Forest
  - SVM
  - Naive Bayes / KNN
- Model evaluation (accuracy, precision, recall, F1-score)
- Confusion matrix and accuracy bar plots
- Save model & vectorizer using `joblib` / `pickle`
- Provide user friendly ui for user for real time prediction of information if it is false or true.

---

## 📂 Dataset

The project uses a CSV file (`train.csv`) containing the following columns:
- `author`
- `title`
- `text`
- `label` → 0 (Real), 1 (Fake)

---

## 🧹 Data Preprocessing

- Merged `author`, `title`, and `text` columns
- Removed non-alphabetic characters
- Converted text to lowercase
- Removed stopwords
- Applied Porter Stemmer
- Vectorized text using **TF-IDF**

---

## 🤖 Models Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Multinomial Naive Bayes / K-Nearest Neighbors (KNN)

Each model is trained and evaluated on test data split using stratified sampling.

---

## 📊 Model Evaluation Metrics

Each model is evaluated using the following:

- ✅ Accuracy
- 🎯 Precision
- 📈 Recall
- 📊 F1-Score
- 📉 Confusion Matrix

---

## 📈 Visualizations

### 🔹 Model Accuracy Comparison
![image](https://github.com/user-attachments/assets/67816ed2-2253-4014-83cd-b4a3b0aed169)



### 🔹 Confusion Matrices

![image](https://github.com/user-attachments/assets/0135429c-6fb3-40d7-904e-29aa42a16394)

---

## 🛠️ How to Run

### 🔧 Prerequisites
```bash
pip install numpy pandas nltk scikit-learn seaborn matplotlib joblib


---

## 🛠️ How to Run

### 🔧 Prerequisites
```bash
pip install numpy pandas nltk scikit-learn seaborn matplotlib joblib

🚀 Steps
Place your dataset as train.csv in the root directory.

Run the notebook : python preprocess_and_train.py

This will:

Preprocess the data

Train all models

Save models and vectorizer as .pkl files

📂 Project Structure
mathematica
Copy
Edit
📁 False-Information-Detection-using-ML
├── train.csv
├── preprocess_and_train.py
├── app.py
├── accuracy_plot.png
├── confusion_matrices.png
├── Logistic Regression.pkl
├── Random Forest.pkl
├── SVM.pkl
├── vectorizer.pkl
└── README.md


💡 Future Enhancements
Integration with a Flask/Streamlit UI

Live fake news classification API

Use of deep learning (LSTM, BERT) for better performance

Model comparison dashboard


🙌 Credits
Dataset: Kaggle Fake News Dataset

Libraries: nltk, scikit-learn, pandas, seaborn, matplotlib, Stramlit


📢 License
This project is open-source and available under the MIT License.
