import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import joblib
import matplotlib.pyplot as plt

# Function for text preprocessing
def preprocess_text(content):
    port_stem = PorterStemmer()
    content = re.sub('[^a-zA-Z]', ' ', str(content))  # Convert NaN to string
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content

# Load the model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('logistic_regression_model.pkl')
    return model, vectorizer

# Function to load the dataset
@st.cache_data
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='latin-1')
    except pd.errors.ParserError as e:
        st.error(f"ParserError: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None
    return data

# Main function
def main():
    st.title('False Information Detection')
    st.write('Enter the text to determine if it is Real or Fake Information.')

    text = st.text_area('Input Text:', '', height=300)
    if st.button('Predict'):
        if text:
            model, vectorizer = load_model_and_vectorizer()
            processed_text = preprocess_text(text)
            X_input = vectorizer.transform([processed_text])
            prediction = model.predict(X_input)[0]

            if prediction == 0:
                st.write('The information is Real')
            else:
                st.write('The information is False')

    st.header('Model Evaluation on Train Dataset')
    train_data = load_dataset('train.csv')

    if train_data is not None:
        st.write("Train dataset loaded successfully.")
        st.write(train_data.head())

        # Remove rows with NaN values in 'text' or 'label' columns
        train_data = train_data.dropna(subset=['text', 'label'])

        if train_data.empty:
            st.error("The dataset is empty after removing rows with NaN values. Please provide a valid dataset.")
            return

        # Preprocess the train data
        train_data['text'] = train_data['text'].apply(preprocess_text)

        X_test = train_data['text'].values
        Y_test = train_data['label'].values

        model, vectorizer = load_model_and_vectorizer()
        Y_pred = model.predict(vectorizer.transform(X_test))

        st.subheader('Confusion Matrix')
        cm = confusion_matrix(Y_test, Y_pred)
        st.write(cm)

        st.subheader('Classification Report')
        report = classification_report(Y_test, Y_pred)
        st.write(report)

        if np.any(Y_pred == 1):
            st.write("Some news were predicted as fake.")
        else:
            st.write("All news were predicted as real.")
    else:
        st.write("Failed to load the train dataset.")

if __name__ == '__main__':
    main()
