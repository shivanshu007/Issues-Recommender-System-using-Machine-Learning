import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel


@st.cache(allow_output_mutation=True)
def load_model():
    tfidf_matrix = joblib.load('model_matrix.pkl')
    tfidf_vectorizer = joblib.load('model_vectorizer.pkl')
    data = pd.read_csv('car_issues_data.csv')
    return tfidf_matrix, tfidf_vectorizer, data


def recommend_issues(description, tfidf_matrix, tfidf_vectorizer, data, top_n=5):
    description_tfidf = tfidf_vectorizer.transform([description])
    cosine_similarities = linear_kernel(description_tfidf, tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[:-top_n - 1:-1]
    recommendations = data.iloc[similar_indices][['issue_description', 'solution']]
    return recommendations

def evaluate_model(test_data, tfidf_matrix, tfidf_vectorizer, data):
    correct = 0
    for idx, row in test_data.iterrows():
        recommendations = recommend_issues(row['issue_description'], tfidf_matrix, tfidf_vectorizer, data)
        if any(test_data.iloc[idx]['solution'] == rec['solution'] for rec in recommendations.to_dict('records')):
            correct += 1
    accuracy = correct / len(test_data)
    return accuracy


st.title("Car Issues Recommender System")
 # Load Model
tfidf_matrix, tfidf_vectorizer, data = load_model()

issue_description = st.text_area("Describe the car issue you are facing:")

# Prepare a test set
test_data = pd.DataFrame({
    'issue_description': [
        "engine makes strange noise when accelerating",
        "car won't start, battery is dead",
        "brakes are making a squeaking noise"
    ],
    'solution': [
        "Inspect CV joints and axles",
        "Replace or recharge the battery",
        "Inspect brake pads and rotors"
    ]
})
# Evaluate the model
accuracy = evaluate_model(test_data, tfidf_matrix, tfidf_vectorizer, data)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

if st.button("Get Recommendations"):
    if issue_description:
        recommendations = recommend_issues(issue_description, tfidf_matrix, tfidf_vectorizer, data)
        st.write("Here are the top recommendations for your issue:")
        for idx, row in recommendations.iterrows():
            st.write(f"Issue: {row['issue_description']}")
            st.write(f"Solution: {row['solution']}")
            st.write("---")

        feedback = st.radio(
            "Was the recommended solution helpful?",
            ("Yes", "No")
        )

        if feedback:
            st.write("Thank you for your feedback!")
    else:
        st.write("Please enter a description of the car issue.")
