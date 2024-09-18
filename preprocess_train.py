# preprocess_and_train.py
import numpy as np
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import nltk

nltk.download('stopwords')

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    news_dataset = pd.read_csv(file_path)
    news_dataset = news_dataset.fillna('')
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

    port_stem = PorterStemmer()
    def stemming(content):
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content

    news_dataset['content'] = news_dataset['content'].apply(stemming)
    return news_dataset

# Train the models and save them
def train_and_save_models(news_dataset):
    X = news_dataset['content'].values
    Y = news_dataset['label'].values

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    for model_name, model in models.items():
        model.fit(X_train, Y_train)
        with open(f'{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    file_path = '/content/train.csv'  # Change this to the path of your dataset
    news_dataset = load_and_preprocess_data(file_path)
    train_and_save_models(news_dataset)
