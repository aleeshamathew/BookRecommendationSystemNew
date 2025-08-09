from flask import Flask, render_template, request
import pickle
from model import get_recommendations

app = Flask(__name__)

# Load the pre-trained model
with open('book_recommender.pkl', 'rb') as f:
    model_data = pickle.load(f)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_form():
    return render_template('input.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    author = request.form['author']
    publisher = request.form['publisher']
    
    recommendations = get_recommendations(title, author, publisher, model_data)
    
    return render_template('output.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)