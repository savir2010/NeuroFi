import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ssl

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
    print(avg_sentiment)
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

# Example usage
if __name__ == "__main__":
    news_url = "https://finance.yahoo.com/news/apple-inc-aapl-own-don-163850433.html"
    
    print("Analyzing stock news article...")
    result = analyze_stock_news_from_url(news_url)
    
    print(f"Title: {result.get('title', 'Unknown')}")
    print(f"URL: {result.get('url', 'Unknown')}")
    
    if "error" in result:
        print(f"Error: {result['error']}")
    elif "message" in result:
        print(f"Message: {result['message']}")
    else:
        print("\nTickers mentioned in the article:")
        for item in result["tickers"]:
            print(f"\n{item['ticker']}: {item['sentiment']} (score: {item['score']:.2f}, mentions: {item['mentions']})")
            if 'contexts' in item:
                print("\nContext examples:")
                for i, context in enumerate(item['contexts']):
                    print(f"  {i+1}. ...{context}...")
    
    print("\nSimplified Results:")
    simplified = get_simplified_stock_sentiment(news_url)
    print(simplified)