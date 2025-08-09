import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def train_and_save_model():
    # Load the dataset
    books_data = pd.read_csv('Books.csv')
    selected_features = ['Book-Title', 'Book-Author', 'Publisher']
    
    # Handle missing values
    for feature in selected_features:
        books_data[feature] = books_data[feature].fillna(' ')
    
    # Combine features
    combined_features = books_data['Book-Title'] + ' ' + books_data['Book-Author'] + ' ' + books_data['Publisher']
    
    # Initialize and fit the vectorizer
    vectorizer = TfidfVectorizer()
    book_vectors = vectorizer.fit_transform(combined_features)
    
    # Save the model and data
    model_data = {
        'vectorizer': vectorizer,
        'book_vectors': book_vectors,
        'books_data': books_data
    }
    
    with open('book_recommender.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

def get_recommendations(title, author, publisher, model_data, n=5):
    # Load model components
    vectorizer = model_data['vectorizer']
    book_vectors = model_data['book_vectors']
    books_data = model_data['books_data']
    
    # Combine user input
    user_input = f"{title} {author} {publisher}"
    
    # Transform user input
    user_vector = vectorizer.transform([user_input])
    
    # Calculate similarities
    similarities = cosine_similarity(user_vector, book_vectors).flatten()
    
    # Get top N similar books
    top_indices = similarities.argsort()[-n:][::-1]
    
    # Return recommendations with additional details
    recommendations = []
    for index in top_indices:
        book_info = {
            'title': books_data.iloc[index]['Book-Title'],
            'author': books_data.iloc[index]['Book-Author'],
            'publisher': books_data.iloc[index]['Publisher']
        }
        recommendations.append(book_info)
    
    return recommendations

if __name__ == "__main__":
    model_data = train_and_save_model()