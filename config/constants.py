# Trading Constants
SUPPORTED_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'BTC', 'ETH', 'SOL', 'ADA', 'DOT',
    'SPY', 'QQQ', 'DIA', 'IWM', 'VTI'
]

CRYPTO_SYMBOLS = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'DOGE', 'XRP', 'LINK']
EQUITY_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
ETF_SYMBOLS = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VTV', 'AGG', 'TLT']

# Time Periods
LOOKBACK_PERIODS = {
    'short': 20,
    'medium': 60,
    'long': 252,
}

# Risk Management
MAX_DRAWDOWN_PERCENT = 25
POSITION_SIZE_PERCENT = 2
RISK_FREE_RATE = 0.04
MIN_SHARPE_RATIO = 1.0

# Model Training
MINIMUM_TRAINING_SAMPLES = 500
MINIMUM_VALIDATION_SAMPLES = 100
MINIMUM_TEST_SAMPLES = 50

# Feature Engineering
TECHNICAL_INDICATORS = [
    'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'rsi_14', 'bb_upper', 'bb_lower', 'atr',
    'macd', 'signal_line', 'histogram',
    'adx', 'cci', 'keltner_upper', 'keltner_lower'
]

VOLATILITY_INDICATORS = [
    'volatility_20d', 'volatility_60d',
    'parkinson_vol', 'garman_klass_vol',
    'rogers_satchell_vol'
]

VOLUME_INDICATORS = [
    'volume_sma', 'volume_ratio',
    'obv', 'adl', 'cmf', 'mfi'
]

# API Rate Limits
POLYGON_RATE_LIMIT = 5  # requests per minute for free tier
COINBASE_RATE_LIMIT = 10
PERPLEXITY_RATE_LIMIT = 3  # requests per minute
DEEPSEEK_RATE_LIMIT = 60  # requests per minute

# Data Validation
MAX_RETURN_PER_DAY = 0.50  # 50% max move detected as outlier
MIN_VOLUME_MULTIPLIER = 0.1  # minimum of mean volume
MAX_PRICE_CHANGE_PCT = 0.25  # warning threshold

# Backtesting
SLIPPAGE_BPS = 2  # basis points
COMMISSION_BPS = 1  # basis points
MINIMUM_HOLDING_PERIOD = 1  # days
REBALANCE_FREQUENCY = 'weekly'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'financial_nn.log'
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_LOG_COUNT = 5

# Database
DB_BATCH_SIZE = 1000
DB_TIMEOUT = 30

# Performance Targets
TARGET_SHARPE_RATIO = 1.5
TARGET_WIN_RATE = 0.55
TARGET_MAX_DRAWDOWN = -0.20
TARGET_CALMAR_RATIO = 2.0

# Model Checkpoint
SAVE_MODEL_EVERY_N_EPOCHS = 10
KEEP_TOP_N_MODELS = 3
