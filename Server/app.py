# Import Flask  
from flask import Flask, render_template, request           
from transformers import pipeline
import numpy as np
import joblib
import imdb
import os
import requests

# Constants for Movie Metadata
BASE_URL = "https://api.themoviedb.org/3"
API_KEY = "fb6cd9a842dd77355df496b80e19bf61"  # Replace with your TMDB API Key

# Load Data
MoviesData = joblib.load('Movies_Datase.pkl')
X = joblib.load('Movies_Learned_Features.pkl')

my_ratings = np.zeros((9724, 1))
my_movies = []
my_added_movies = []




def computeCost(X, y, theta):
    m = y.size
    s = np.dot(X, theta) - y
    j = (1 / (2 * m)) * (np.dot(np.transpose(s), s))
    print(j)
    return j

def gradientDescent(X, y, theta, alpha, num_iters):  
    m = float(y.shape[0])
    theta = theta.copy()
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(np.transpose((np.dot(X, theta) - y)), X))
    return theta

def checkAndAdd(movie, rating):
    try:
        if isinstance(int(rating), str):
            pass
    except ValueError:
        return 3
    if 0 <= int(rating) <= 5:
        movie = movie.lower()
        movie = movie + ' '
        if movie not in MoviesData['title'].unique():
            return 1
        else:
            index = MoviesData[MoviesData['title'] == movie].index.values[0]
            my_ratings[index] = rating
            movieid = MoviesData.loc[MoviesData['title'] == movie, 'movieid']
            if movie in my_added_movies:
                return 2
            my_movies.append(movieid)
            my_added_movies.append(movie)
            return 0
    else:
        return -1

def url_clean(url):
    base, ext = os.path.splitext(url)
    i = url.count('@')
    s2 = url.split('@')[0]
    url = s2 + '@' * i + ext
    return url

# Fetch Movie Metadata including Trailer
def fetch_movie_details(movie_title):
    try:
        # Search for the movie by title
        search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={movie_title}"
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_results = search_response.json().get('results', [])

        if not search_results:
            return None  # No movie found

        # Use the first result for the movie
        movie_id = search_results[0]['id']
        
        # Fetch movie details including credits and reviews
        details_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&append_to_response=credits,videos,reviews"
        details_response = requests.get(details_url)
        details_response.raise_for_status()
        details = details_response.json()

        # Extract required details
        movie_data = {
            "title": details.get('title', 'N/A'),
            "summary": details.get('overview', 'No summary available'),
            "poster_url": f"https://image.tmdb.org/t/p/w500{details.get('poster_path', '')}",
            "rating": details.get('vote_average', 'N/A'),
            "genres": [genre['name'] for genre in details.get('genres', [])],
            "release_date": details.get('release_date', 'N/A'),
            "cast": [cast['name'] for cast in details.get('credits', {}).get('cast', [])[:5]],  # Top 5 cast members
        }

        # Extract trailer URL if available
        videos = details.get('videos', {}).get('results', [])
        if videos:
            # Look for the first YouTube trailer (if available)
            trailer = next((video for video in videos if video['site'] == 'YouTube'), None)
            if trailer:
                movie_data["trailer_url"] = f"https://www.youtube.com/embed/{trailer['key']}"
            else:
                movie_data["trailer_url"] = None
        else:
            movie_data["trailer_url"] = None

        # Get Reviews
        reviews = details.get('reviews', {}).get('results', [])
        movie_data["reviews"] = [review['content'] for review in reviews]

        return movie_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie details: {e}")
        return None


# Load the pre-trained model from Hugging Face (for sentiment analysis)
sentiment_pipeline = pipeline("sentiment-analysis",  model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',truncation=True, max_length=512)

def analyze_sentiment(reviews):
    sentiment_results = {"positive": 0, "neutral": 0, "negative": 0}
    review_sentiments = []

    for review in reviews:
        # Perform sentiment analysis using BERT
        sentiment = sentiment_pipeline(review)[0]
        label = sentiment['label']
        score = sentiment['score']

        # Classify sentiment based on BERT output
        if label == 'POSITIVE':
            sentiment_label = "Positive"
            sentiment_results["positive"] += 1
        elif label == 'NEGATIVE':
            sentiment_label = "Negative"
            sentiment_results["negative"] += 1
        else:
            sentiment_label = "Neutral"
            sentiment_results["neutral"] += 1

        # Add the sentiment label and review text to the list
        review_sentiments.append({
            "review": review,
            "sentiment": sentiment_label,
            "score": score  # Add the confidence score from the model
        })

    total_reviews = len(reviews)
    if total_reviews > 0:
        # Calculate percentages
        sentiment_results["positive_percent"] = (sentiment_results["positive"] / total_reviews) * 100
        sentiment_results["negative_percent"] = (sentiment_results["negative"] / total_reviews) * 100
        sentiment_results["neutral_percent"] = (sentiment_results["neutral"] / total_reviews) * 100
    else:
        sentiment_results["positive_percent"] = sentiment_results["negative_percent"] = sentiment_results["neutral_percent"] = 0

    return review_sentiments, sentiment_results


# Initialize the BART model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def generate_summary(movie_summary):
    # Check if the summary is not empty
    if movie_summary:
        # Generate the summary using the BART model
        summary = summarizer(movie_summary, max_length=70, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    return "No summary available to summarize."


# Flask App Setup
app = Flask(__name__)

@app.route('/movieDetails/<movie_title>', methods=['GET'])
def movie_details(movie_title):
    movie_data = fetch_movie_details(movie_title)
    if not movie_data:
        return render_template('error.html', message="Movie details not found!")

    # Perform sentiment analysis on the reviews
    review_sentiments, sentiment_results = analyze_sentiment(movie_data.get("reviews", []))

    # Generate a summary of the movie's plot using BART
    generated_summary = generate_summary(movie_data.get("summary", ""))

    return render_template('movie_details.html', 
                           movie=movie_data, 
                           sentiment=sentiment_results, 
                           review_sentiments=review_sentiments,
                           generated_summary=generated_summary)  # Add this line



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/addMovie/', methods=['GET', 'POST'])
def addMovie():
    val = request.form.get('movie_name')
    rating = request.form.get('rating')
    flag = checkAndAdd(val, rating)
    if flag == 1:
        processed_text = "Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies"
        return render_template('home.html', processed_text=processed_text)
    elif flag == -1:
        processed_text = "Please enter rating between 1-5. This application follows a five-star rating system"
        return render_template('home.html', processed_text=processed_text)
    elif flag == 2:
        processed_text = "The movie has already been added by you"
        return render_template('home.html', processed_text=processed_text)
    elif flag == 3:
        processed_text = "Invalid Input! Please enter a number between 0-5 in the rating field"
        return render_template('home.html', processed_text=processed_text)
    else:        
        processed_text = "Successfully added movie to your rated movies"
        movie_text = ", you've rated " + rating + " stars to movie: " + val
        return render_template('home.html', processed_text=processed_text, movie_text=movie_text, my_added_movies=my_added_movies)

@app.route('/reset/', methods=['GET', 'POST'])
def reset():
    global my_ratings
    global my_movies
    global my_added_movies
    
    my_ratings = np.zeros((9724, 1))
    my_movies = []
    my_added_movies = []
    processed_text = 'Successfully reset'
    return render_template('home.html', processed_text=processed_text)

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        if len(my_added_movies) == 0:
            processed_text = "Yikes! You've to add some movies before predicting anything"
            return render_template('home.html', processed_text=processed_text)
        
        out_arr = my_ratings[np.nonzero(my_ratings)]
        out_arr = out_arr.reshape(-1, 1)
        idx = np.where(my_ratings)[0]
        X_1 = [X[x] for x in idx]
        X_1 = np.array(X_1)
        y = out_arr.flatten()
        theta = gradientDescent(X_1, y, np.zeros((100)), 0.001, 4000)
        
        p = X @ theta.T
        p = p.flatten()
        
        predictedData = MoviesData.copy()
        predictedData['Prediction'] = p
        sorted_data = predictedData.sort_values(by=['Prediction'], ascending=False)
        sorted_data = sorted_data[~sorted_data.title.isin(my_added_movies)].iloc[:40]
        
        recommendations = []
        for _, row in sorted_data.iterrows():
            movie_title = row['title']
            try:
                metadata = fetch_movie_details(movie_title)
                poster_url = metadata.get('poster_url', 'N/A')
            except Exception as e:
                print(f"Error fetching metadata for {movie_title}: {e}")
                poster_url = "N/A"
            recommendations.append([movie_title, row['Prediction'], poster_url])
        
        return render_template('result.html', my_list=recommendations)

if __name__ == '__main__':
    app.run()