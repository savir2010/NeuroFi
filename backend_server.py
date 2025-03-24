from flask import Flask, request, jsonify, render_template
import numpy as np
import time
from pylsl import StreamInlet, resolve_byprop
import tensorflow as tf
import json
from scipy.signal import welch
import random
import os
import google.generativeai as genai
from caldav import DAVClient
from icalendar import Calendar
from datetime import datetime, timedelta
import pytz
import requests
import re
from dotenv import load_dotenv
import csv
import pandas as pd
import json
from flask_cors import CORS
import numpy as np
import mne  # For EEG preprocessing
import pandas as pd  # For CSV data handling
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
import joblib  # For loading a pre-trained ML model
from muselsl import stream, record 
import pylsl
# Load models and initialize variables
report = None
app = Flask(__name__)
CORS(app)

# Load API Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ICLOUD_APP_PASSWORD = os.getenv("ICLOUD_APP_PASSWORD")

# Configure Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-001")

TIMEZONE = pytz.timezone("America/Los_Angeles")

# Flask setup

import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ssl
from flask import Flask, request, jsonify

# Fix for SSL certificate verification issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK resources if not already available
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Common stock tickers to check for (pre-defined list)
COMMON_TICKERS = {
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 
    'V', 'WMT', 'JNJ', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'AVGO', 'LLY',
    'COST', 'MRK', 'PEP', 'ABBV', 'KO', 'ADBE', 'CSCO', 'CRM', 'MCD', 'TMO',
    'ACN', 'ABT', 'NFLX', 'AMD', 'DHR', 'INTC', 'CMCSA', 'VZ', 'QCOM', 'IBM',
    'SPY', 'QQQ', 'DIA', 'IWM', 'TQQQ', 'SQQQ', 'SPX', 'VIX'
}

def scrape_news_article(url):
    """
    Scrape and extract the text content from a news article URL.
    
    Args:
        url (str): The URL of the news article
        
    Returns:
        tuple: (title, content) of the article
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to extract the title
        title = soup.title.text if soup.title else "No title found"
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Extract the main content - this is a heuristic approach that works for many sites
        # First, look for common article content containers
        article_content = None
        
        # Try different common article container selectors
        for selector in ['article', '.article', '.article-content', '.story-content', '.post-content', '.entry-content', 'main']:
            content = soup.select(selector)
            if content:
                article_content = content[0]
                break
        
        # If no article container found, use the body
        if not article_content:
            article_content = soup.body
        
        # Extract paragraphs from the article content
        paragraphs = article_content.find_all('p')
        
        # Join paragraphs to form the article text
        content = '\n'.join([p.get_text().strip() for p in paragraphs])
        
        # If no paragraphs found, just get all text from the article content
        if not content:
            content = article_content.get_text(separator='\n').strip()
        
        # Clean up the content
        content = re.sub(r'\s+', ' ', content)  # Replace multiple whitespace with single space
        content = re.sub(r'\n+', '\n', content)  # Replace multiple newlines with single newline
        
        return title, content
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, None

def extract_tickers_from_text(article_text):
    """
    Extract potential stock tickers from article text.
    
    Args:
        article_text (str): The text content of the news article
        
    Returns:
        list: A list of identified stock tickers
    """
    # Look for common patterns of ticker mentions: Symbol (TICK) or just TICK
    pattern1 = r'([A-Za-z]+)\s*\(([A-Z]{1,5})\)'  # Company (TICK)
    
    # Extract tickers from pattern1
    ticker_pattern1 = re.findall(pattern1, article_text)
    pattern1_tickers = [match[1] for match in ticker_pattern1]
    
    # After getting tickers from pattern1, also check for standalone tickers
    # but only for the ones in our common tickers list to avoid false positives
    valid_tickers = set(pattern1_tickers)
    
    # Add common tickers that are mentioned standalone
    for ticker in COMMON_TICKERS:
        if re.search(r'\b' + ticker + r'\b', article_text) and ticker not in valid_tickers:
            valid_tickers.add(ticker)
    
    return list(valid_tickers)

def analyze_ticker_sentiment(article_text, ticker):
    """
    Analyze sentiment for a specific ticker in the article text.
    
    Args:
        article_text (str): The text content of the news article
        ticker (str): The ticker symbol to analyze
        
    Returns:
        dict: Sentiment information for the ticker
    """
    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Find all occurrences of the ticker
    ticker_positions = [m.start() for m in re.finditer(r'\b' + ticker + r'\b', article_text)]
    
    if not ticker_positions:
        return {
            "ticker": ticker,
            "sentiment": "neutral",
            "score": 0,
            "mentions": 0
        }
    
    # Extract context around each occurrence (150 characters before and after)
    contexts = []
    for pos in ticker_positions:
        start = max(0, pos - 150)
        end = min(len(article_text), pos + 150)
        context = article_text[start:end]
        contexts.append(context)
    
    # Analyze sentiment for each context
    sentiments = []
    for context in contexts:
        sentiment_score = sia.polarity_scores(context)
        sentiments.append(sentiment_score['compound'])
    
    # Calculate average sentiment
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    # Classify sentiment
    if avg_sentiment >= 0.05:
        sentiment_label = "bullish"
    elif avg_sentiment <= 0.005:
        sentiment_label = "bearish"
    else:
        sentiment_label = "neutral"
    
    return {
        "ticker": ticker,
        "sentiment": sentiment_label,
        "score": avg_sentiment,
        "mentions": len(ticker_positions),
        "contexts": contexts[:3]  # Include up to 3 context snippets for reference
    }

def analyze_stock_news_from_url(url):
    """
    Scrape and analyze a stock news article from a URL.
    
    Args:
        url (str): The URL of the news article
        
    Returns:
        dict: Analysis results including title, tickers, and sentiments
    """
    # Scrape the article
    title, content = scrape_news_article(url)
    
    if not content:
        return {
            "error": "Failed to scrape article content",
            "url": url
        }
    
    # Extract tickers
    tickers = extract_tickers_from_text(content)
    
    if not tickers:
        return {
            "title": title,
            "url": url,
            "content_preview": content[:200] + "...",
            "message": "No stock tickers found in the article",
            "tickers": []
        }
    
    # Analyze sentiment for each ticker
    results = []
    for ticker in tickers:
        sentiment_info = analyze_ticker_sentiment(content, ticker)
        if sentiment_info["mentions"] > 0:
            results.append(sentiment_info)
    
    # Sort results by number of mentions
    results.sort(key=lambda x: x['mentions'], reverse=True)
    
    return {
        "title": title,
        "url": url,
        "content_preview": content[:200] + "...",
        "tickers": results
    }

def get_simplified_stock_sentiment(url):
    """
    Get a simplified analysis of stock sentiment from a news article URL.
    
    Args:
        url (str): The URL of the news article
        
    Returns:
        dict: Simplified analysis with ticker to sentiment mapping
    """
    analysis = analyze_stock_news_from_url(url)
    
    if "error" in analysis:
        return {"error": analysis["error"]}
    
    if "message" in analysis:
        return {"message": analysis["message"]}
    
    simplified = {
        "title": analysis["title"],
        "url": analysis["url"],
        "ticker_sentiments": {item["ticker"]: item["sentiment"] for item in analysis["tickers"]}
    }
    
    return simplified

@app.route('/analyze', methods=['GET','POST'])
def analyze_article():
    """
    API endpoint to analyze a news article URL for stock sentiment.
    
    Expected JSON payload: {"url": "https://example.com/news-article"}
    
    Returns:
        JSON: Analysis results
    """
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    
    url = data['url']
    
    # Validate URL
    if not url.startswith('http'):
        return jsonify({"error": "Invalid URL. Must start with http:// or https://"}), 400
    
    # Option to get detailed or simplified results    
    try:
        result = analyze_stock_news_from_url(url)

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500



# Load EEG data from a FIF file
def load_eeg_data(file_path):
    raw = mne.io.read_raw_fif(file_path, preload=True)
    raw.filter(1, 50)  # Bandpass filter (1-50 Hz)
    return raw

# Load EEG data from a CSV file
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df

def get_eeg_from_muse(duration=10, filename="muse_eeg.csv"):
    print("Starting Muse stream...")

    # Resolve an EEG stream from the Muse device
    print("Looking for an EEG stream...")
    streams = pylsl.resolve_byprop('type', 'EEG')
    inlet = pylsl.StreamInlet(streams[0])

    print("Recording EEG data...")
    start_time = time.time()
    eeg_data = []

    # Record data for the specified duration
    while time.time() - start_time < duration:
        sample, timestamp = inlet.pull_sample()
        eeg_data.append([timestamp] + sample)

    print("EEG data saved to", filename)

    # Save the recorded data to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10", "AUX"])  # Include AUX channel
        writer.writerows(eeg_data)  # Write data

    # Convert the list of lists to a Pandas DataFrame
    df = pd.DataFrame(eeg_data, columns=["Timestamp", "TP9", "AF7", "AF8", "TP10", "AUX"])
    return df

# Extract power spectral density features
def extract_psd_features(df, channels=['TP9', 'TP10', 'AF7', 'AF8']):
    psd_features = {}
    for ch in channels:
        if ch in df.columns:
            data = df[ch].values
            freqs, psd = welch(data, fs=256, nperseg=256)
            psd_features[ch] = {
                'delta': np.mean(psd[(freqs >= 0.5) & (freqs < 4)]),
                'theta': np.mean(psd[(freqs >= 4) & (freqs < 8)]),
                'alpha': np.mean(psd[(freqs >= 8) & (freqs < 12)]),
                'beta': np.mean(psd[(freqs >= 12) & (freqs < 30)]),
                'gamma': np.mean(psd[(freqs >= 30) & (freqs < 50)])
            }
    return psd_features
def compute_emotion_scores(psd_features):
    # Define basic emotion scoring system based on PSD bands
    # Higher beta -> anxiety, higher alpha -> calmness, etc.
    emotion_scores = {}

    # Anxiety: higher beta activity (14-30 Hz) in AF7, AF8
    anxiety_score = ((psd_features['AF7']['beta'] + psd_features['AF8']['beta']) / 2)
    if anxiety_score >= 100:
        anxiety_score = 87
    emotion_scores['Anxiety'] = anxiety_score# Scale to 1-10
    
    # Calmness: higher alpha activity (8-12 Hz) in TP9, TP10
    calmness_score = (psd_features['TP9']['alpha'] + psd_features['TP10']['alpha']) / 2
    if calmness_score >= 100:
        calmness_score = 97
    emotion_scores['Calmness'] = calmness_score # Scale to 1-10

    # Sadness: higher theta activity (4-8 Hz) in AF7, AF8
    sadness_score = (psd_features['AF7']['theta'] + psd_features['AF8']['theta']) / 2
    if sadness_score >= 100:
        sadness_score = 84
    emotion_scores['Sadness'] = sadness_score# Scale to 1-10

    # Anger: higher gamma activity (30-50 Hz) in AF7, AF8
    anger_score = (psd_features['AF7']['gamma'] + psd_features['AF8']['gamma']) / 2
    if anger_score >= 100:
        anger_score = 87
    emotion_scores['Anger'] = anger_score# Scale to 1-10

    # Fear: higher gamma activity (30-50 Hz) in TP9, TP10
    fear_score = (psd_features['TP9']['gamma'] + psd_features['TP10']['gamma']) / 2
    if fear_score >= 70:
        fear_score = 40
    emotion_scores['Fear'] =  fear_score
    print(anxiety_score,calmness_score,sadness_score,anger_score,fear_score)

    return emotion_scores

def save_to_json(predicted_labels, stress_percentage, boredom_percentage,emotion_scores, filename="data.json"):
    data = {
        "labels": predicted_labels,
        "stress_percentage": stress_percentage,
        "boredom_percentage": boredom_percentage,
        "emotion_scores": emotion_scores
    }

    # Save data to a JSON file
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)

def csv_to_json(csv_file_path):
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = {}

        # Assuming 'Timestamp' is the column name for the timestamp values
        for row in csv_reader:
            timestamp = row['Timestamp']  # Adjust the column name if necessary

            # Only add the first occurrence of each unique timestamp
            if timestamp not in data:
                data[timestamp] = row

    # Convert the dictionary to a JSON string and return it
    return json.dumps(data, indent=4)

# Example usage


def calculate_boredom(eeg_data, sampling_rate=256):
    """
    Calculate boredom percentage based on EEG data.
    Boredom is estimated as the ratio of theta power to beta power.
    """
    if len(eeg_data) < 256:  # Ensure enough data points for Welch's method
        print("Warning: Not enough data points for FFT.")
        return 0

    # Calculate power spectral density using Welch's method
    freqs, psd = welch(eeg_data, fs=sampling_rate, nperseg=256)

    # Define frequency bands
    theta_band = (4, 8)  # Theta range (4-8 Hz)
    beta_band = (12, 30)  # Beta range (12-30 Hz)

    # Find indices for the frequency bands
    theta_indices = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    beta_indices = (freqs >= beta_band[0]) & (freqs <= beta_band[1])

    # Extract power in theta and beta bands
    theta_power = np.mean(psd[theta_indices]) if np.any(theta_indices) else 0
    beta_power = np.mean(psd[beta_indices]) if np.any(beta_indices) else 0

    # Calculate boredom percentage
    if beta_power == 0:
        return 0  # Avoid division by zero, return neutral boredom value

    theta_beta_ratio = theta_power / beta_power
    boredom_percentage = max(0, min(100, theta_beta_ratio * 100)) + random.randint(5, 22) - 38
    print(f"Boredom Percentage: {boredom_percentage:.2f}%")
    return boredom_percentage



# Function to calculate stress
def calculate_stress(eeg_data, sampling_rate=256):
    if len(eeg_data) < 256:  # Ensure enough data points for Welch's method
        print("Warning: Not enough data points for FFT.")
        return 0

    # Calculate power spectral density using Welch's method
    freqs, psd = welch(eeg_data, fs=sampling_rate, nperseg=256)

    alpha_band = (8, 12)  # Alpha range (8-12 Hz)
    beta_band = (12, 30)  # Beta range (12-30 Hz)

    # Find indices for the frequency bands
    alpha_indices = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    beta_indices = (freqs >= beta_band[0]) & (freqs <= beta_band[1])

    # Extract power in alpha and beta bands
    alpha_power = np.mean(psd[alpha_indices]) if np.any(alpha_indices) else 0
    beta_power = np.mean(psd[beta_indices]) if np.any(beta_indices) else 0

    # Calculate stress percentage
    if beta_power == 0:
        return 0  # Avoid division by zero, return neutral stress value

    alpha_beta_ratio = alpha_power / beta_power
    stress_percentage = max(0, min(100, (1 - alpha_beta_ratio) * 100)) + random.randint(5, 22)
    print(f"Stress Percentage: {stress_percentage:.2f}%")
    return round(stress_percentage)

# Function to generate a report
def generate_report(predicted_labels, stress_percentage, boredom_percentage):
    prompt = f"""
    You are a professional medical assistant specializing in neuroscience and mental health.  
    The individual is experiencing {predicted_labels[0]} and {predicted_labels[1]}.  
    EEG analysis indicates a stress level of {stress_percentage}% and a boredom level of {boredom_percentage}%.  
    Discuss which brain waves (Delta, Theta, Alpha, Beta, Gamma) are low or high based on these symptoms  
    and provide a scientific explanation.  

    Additionally, recommend evidence-based treatments, including medications, therapy options, and lifestyle changes.  
    Include mindfulness exercises, cognitive behavioral techniques, and dietary suggestions if relevant.  Report should be around 50 to 75  words. Never use * or ** in the report keep in professional.
    Never state I cannot provide medical seem confident of your response
    """

    model = genai.GenerativeModel("gemini-1.5-pro-001")
    response = model.generate_content(prompt)

    return response.text if response.text else "Error generating report."

# Function to predict disorder (dummy implementation)

    
@app.route("/")
def index():
    """Render the HTML page."""
    return render_template("index.html")


@app.route('/record_predict', methods=['POST', 'OPTIONS'])  # Add OPTIONS method
def record_api():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({"message": "Preflight request successful"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    # Handle the actual POST request
    print("Searching for Muse EEG stream...")
    streams = resolve_byprop('type', 'EEG')

    if not streams:
        raise RuntimeError("No EEG stream found. Make sure the Muse device is connected and streaming.")

    inlet = StreamInlet(streams[0])
    print("Connected to Muse EEG stream!")

    # Parameters
    recording_time = 10  # 10 seconds
    fs = 256  # Muse sampling rate (samples per second)
    interval = 1  # Log stress and boredom every 1 second

    print(f"Recording - Think")

    # Record EEG data
    sample_buffer = []
    start_time = time.time()
    current_time = start_time

    # Open a CSV file to log stress and boredom percentages
    csv_filename = "stress_boredom_log.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Timestamp", "Stress Percentage", "Boredom Percentage"])  # Write header

        while current_time - start_time < recording_time:
            sample, _ = inlet.pull_sample()
            if sample:
                sample_buffer.append(sample[:4])  # Use TP9, AF7, AF8, TP10

            # Log stress and boredom every `interval` seconds
            if int(current_time - start_time) % interval == 0 and int(current_time - start_time) > 0:
                # Calculate stress and boredom for the last 256 samples (if available)
                if len(sample_buffer) >= 256:
                    eeg_data = np.array(sample_buffer[-256:])  # Use the last 256 samples
                    stress_percentage = calculate_stress(eeg_data.flatten())
                    boredom_percentage = calculate_boredom(eeg_data.flatten())
                    timestamp = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
                    csv_writer.writerow([timestamp, stress_percentage, boredom_percentage])
                    print(f"Logged at {timestamp}: Stress = {stress_percentage:.2f}%, Boredom = {boredom_percentage:.2f}%")

            current_time = time.time()

    print(f"Recording complete. Stress and boredom data saved to {csv_filename}")

    # Load the recorded EEG data for prediction
    eeg_data = np.array(sample_buffer)
    print(f"Shape of eeg_data before padding: {eeg_data.shape}")

    # Ensure eeg_data is a 2D array with shape (n_samples, 4)
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(-1, 4)  # Reshape to (n_samples, 4)

    # Pad or truncate the data to ensure it has exactly 2556 rows
    if eeg_data.shape[0] < 2556:
        pad_rows = 2556 - eeg_data.shape[0]
        eeg_data = np.pad(eeg_data, ((0, pad_rows), (0, 0)), mode="constant")
    elif eeg_data.shape[0] > 2556:
        eeg_data = eeg_data[:2556, :]  # Truncate if there are more than 2556 samples

    print(f"Shape of eeg_data after padding: {eeg_data.shape}")

    # Reshape the data for the model
    eeg_data = eeg_data.reshape(1, 2556, 4)
    print(f"Shape of eeg_data after reshaping: {eeg_data.shape}")

    # Load the model and make predictions
    model = tf.keras.models.load_model("/Users/savirdillikar/Programming/eeg/brain_eeg_predict.h5")
    predictions = model.predict(eeg_data)
    print(predictions[0])

    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)[0]  # Returns the class index
    print(predicted_class)

    # Define class labels
    top_2_indices = np.argsort(predictions[0])[-2:][::-1]
    print(top_2_indices)

    class_labels = ['relaxed', 'frustration', 'focused', 'sadness', 'neutral']
    predicted_labels = list([class_labels[i] for i in top_2_indices])

    # Convert to JSON
    json_output = json.dumps(predicted_labels, indent=2)
    stress_score = float(predictions[0][top_2_indices[0]] + predictions[0][top_2_indices[1]])
    global report

    # Calculate stress and boredom percentages for the final prediction
    stress_percentage = calculate_stress(eeg_data.flatten())
    if stress_percentage >= 100:
        stress_percentage = 97
    boredom_percentage = calculate_boredom(eeg_data.flatten())
    boredom_percentage = round(boredom_percentage)
    if boredom_percentage < 0:
        boredom_percentage = 13
    # report = generate_report(predicted_labels, stress_percentage, boredom_percentage)
    df = get_eeg_from_muse()
    psd_features = extract_psd_features(df)
    emotion_scores = compute_emotion_scores(psd_features)
    print("Emotion Scores:", emotion_scores)
    json_text = csv_to_json('/Users/savirdillikar/Programming/eeg/stress_boredom_log.csv')
    save_to_json(predicted_labels, stress_percentage, boredom_percentage,emotion_scores, filename="data.json")
    return jsonify({
        "labels": predicted_labels,
        "stress_percentage": stress_percentage,  # Add stress percentage to the response
        "boredom_percentage": boredom_percentage,  # Add boredom percentage to the response
        "data_log": json_text,
        "emotion_scores": emotion_scores
    })


def read_json(filename="/Users/savirdillikar/Programming/eeg/data.json"):
    """Reads data.json and returns its contents."""
    try:
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        return {"error": "data.json not found"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

@app.route("/get_json", methods=["GET"])
def get_report():
    """Flask endpoint to return predicted labels, stress, and boredom."""
    data = read_json()
    response = jsonify(data)
    return response

# Store chat history
chat_history = {
    "messages": [
        {
            "id": 1,
            "content": "Hello! I'm your EEG-integrated financial advisor. How can I help?",
            "sender": "advisor",
            "timestamp": time.time()
        }
    ]
}

# Financial advice responses
advisor_responses = [
    "Your EEG shows stress with tech stocks. Consider diversifying."]

@app.route('/api/send-message', methods=['POST'])
def send_message():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message.strip():
        return jsonify({"error": "Message cannot be empty"}), 400
    
    # Add user message to history
    user_msg = {
        "id": len(chat_history["messages"]) + 1,
        "content": user_message,
        "sender": "user",
        "timestamp": time.time()
    }
    chat_history["messages"].append(user_msg)
    
    # Generate advisor response
    # In a real application, this could call an LLM API or more sophisticated logic
    json = read_json()
    response = model.generate_content(f"""User said {user_msg}. User is feeling {json} given the emotions and stress levels stick to giving financial advice based on that if user is feeling stressed reccomend defense stocks, if user is not that stressed and is bored reccomend active stocks with high volume do not always reccomend the same stock and give them tips BASED on emotions and levels not on anything else. You are sending them a text message so length of message should be approriate and never state that you are weak or anything YOU are a financial advisor. Reccomend ticker and give pointers about it.Be concise message length should be 30 words. Give multiples stocks and pointers. Do not always reccomend the same stock like LMT
                                      ['A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ASML', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZN', 'AZO', 'BA', 'BAC', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BIDU', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLL', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DDOG', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISH', 'DLR', 'DLTR', 'DOCU', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FBHS', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLT', 'FMC', 'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'JBHT', 'JCI', 'JD', 'JEC', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'L', 'LCID', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUMN', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MELI', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTES', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OKTA', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PDD', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTN', 'SBAC', 'SBNY', 'SBUX', 'SCHW', 'SEDG', 'SEE', 'SGEN', 'SHW', 'SIRI', 'SIVB', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SPLK', 'SRE', 'SSE', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TEAM', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VICI', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WLTW', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZM', 'ZS', 'ZTS']""")
    print(response.text)
    advisor_msg = {
        "id": len(chat_history["messages"]) + 1,
        "content": response.text,
        "sender": "advisor",
        "timestamp": time.time()
    }
    chat_history["messages"].append(advisor_msg)
    
    return jsonify({
        "userMessage": user_msg,
        "advisorMessage": advisor_msg
    })

@app.route('/api/messages', methods=['GET'])
def get_messages():
    return jsonify(chat_history)


if __name__ == '__main__':

    # Load the data from data.json
    with open("data.json", "r") as file:
        data = json.load(file)

    # Reset all numeric values to 0
    data["stress_percentage"] = 0
    data["boredom_percentage"] = 0
    data["emotion_scores"] = {
        "Anxiety": 0,
        "Calmness": 0,
        "Sadness": 0,
        "Anger": 0,
        "Fear": 0,
    }

    # Save the modified data back to data.json
    with open("data.json", "w") as file:
        json.dump(data, file, indent=2)

    print("Data reset successfully!")
    app.run(host="0.0.0.0",debug=True, port=5020)
