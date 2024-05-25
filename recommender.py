import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['issue_description'])
    return tfidf_matrix, tfidf_vectorizer


def save_model(tfidf_matrix, tfidf_vectorizer, data, tfidf_matrix_filepath, vectorizer_filepath, data_filepath):
    joblib.dump(tfidf_matrix, tfidf_matrix_filepath)
    joblib.dump(tfidf_vectorizer, vectorizer_filepath)
    data.to_csv(data_filepath, index=False)


if __name__ == "__main__":
    data = load_data('car_issues.csv')
    tfidf_matrix, tfidf_vectorizer = preprocess_data(data)

    # Save the model and data
    save_model(tfidf_matrix, tfidf_vectorizer, data,
               'model_matrix.pkl',
               'Issues-Recommender-System-using-Machine-Learning/model_vectorizer.pkl',
               'Issues-Recommender-System-using-Machine-Learning/car_issues_data.csv')

    print("Model and data saved successfully.")