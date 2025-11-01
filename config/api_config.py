import os
from dotenv import load_dotenv

load_dotenv()

class APIConfig:
    # Polygon.io Configuration
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
    POLYGON_FLAT_FILES_ACCESS_KEY = os.getenv('POLYGON_FLAT_FILES_ACCESS_KEY', '')
    POLYGON_FLAT_FILES_SECRET_KEY = os.getenv('POLYGON_FLAT_FILES_SECRET_KEY', '')
    POLYGON_API_BASE = 'https://api.polygon.io'

    # Coinbase Configuration
    COINBASE_API_KEY = os.getenv('COINBASE_API_KEY', '')
    COINBASE_SECRET_KEY = os.getenv('COINBASE_SECRET_KEY', '')
    COINBASE_API_BASE = 'https://api.coinbase.com'

    # Massive.com Configuration
    MASSIVE_API_KEY = os.getenv('MASSIVE_API_KEY', '')
    MASSIVE_API_BASE = os.getenv('MASSIVE_API_BASE', 'https://api.massive.com')
    MASSIVE_MCP_ENDPOINT = os.getenv('MASSIVE_MCP_ENDPOINT', 'http://localhost:3000')

    # Alternative APIs
    PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    NEWS_SENTIMENT_API_KEY = os.getenv('NEWS_SENTIMENT_API_KEY', '')
    ECONOMIC_CALENDAR_API_KEY = os.getenv('ECONOMIC_CALENDAR_API_KEY', '')

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 300))
    RATE_LIMIT_PER_SECOND = int(os.getenv('RATE_LIMIT_PER_SECOND', 10))
    REQUEST_TIMEOUT = 30
    STREAMING_TIMEOUT = 300
    REALTIME_UPDATE_INTERVAL = 5
    DAILY_UPDATE_TIME = '09:30'
    CACHE_TTL = 3600

    # API Endpoints
    ENDPOINTS = {
        'quote': '/quote',
        'historical': '/historical',
        'streaming': '/streaming',
        'technical_indicators': '/indicators',
        'fundamentals': '/fundamentals',
        'polygon_quote': '/v3/quotes/latest',
        'polygon_aggs': '/v2/aggs/ticker',
    }

config = APIConfig()
