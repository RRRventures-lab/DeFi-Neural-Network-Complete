# DeFi Neural Network Trading Dashboard API

## Overview

FastAPI backend server for the trading dashboard with real-time updates, performance analytics, and model management.

**Base URL**: `http://localhost:8000`
**WebSocket URL**: `ws://localhost:8000/ws/updates`
**API Documentation**: `http://localhost:8000/docs` (Swagger UI)
**Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start the Server

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

## API Endpoints

### Health & Status

#### GET `/`
Root endpoint - API is running
```bash
curl http://localhost:8000/
```

#### GET `/health`
Health check endpoint
```bash
curl http://localhost:8000/health
```

### Authentication

#### POST `/auth/login`
Login with admin password
```bash
curl -X POST "http://localhost:8000/auth/login?password=admin"
```

Response:
```json
{
  "access_token": "token_string",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### POST `/auth/verify`
Verify authentication token
```bash
curl -X POST "http://localhost:8000/auth/verify" \
  -H "Authorization: Bearer token_string"
```

### Trading Operations

#### GET `/api/trading/status`
Get engine status
```bash
curl http://localhost:8000/api/trading/status
```

#### POST `/api/trading/start`
Start trading engine
```bash
curl -X POST "http://localhost:8000/api/trading/start?mode=paper"
```

**Parameters:**
- `mode` (string): "paper" or "live" (default: "paper")

#### POST `/api/trading/stop`
Stop trading engine
```bash
curl -X POST http://localhost:8000/api/trading/stop
```

#### GET `/api/trading/positions`
Get open positions
```bash
curl http://localhost:8000/api/trading/positions
```

#### POST `/api/trading/signal`
Process trading signal
```bash
curl -X POST "http://localhost:8000/api/trading/signal?symbol=BTC&signal_type=buy&strength=0.8"
```

**Parameters:**
- `symbol` (string): Asset symbol
- `signal_type` (string): "buy", "sell", or "hold"
- `strength` (float): 0-1
- `metadata` (object, optional): Additional metadata

#### POST `/api/trading/order`
Create and execute order
```bash
curl -X POST "http://localhost:8000/api/trading/order?symbol=BTC&side=buy&quantity=0.5&order_type=market"
```

**Parameters:**
- `symbol` (string): Asset symbol
- `side` (string): "buy" or "sell"
- `quantity` (float): Order quantity
- `order_type` (string): "market", "limit", "stop"
- `price` (float, optional): For limit/stop orders

#### GET `/api/trading/orders`
Get order history
```bash
curl "http://localhost:8000/api/trading/orders?limit=100"
```

#### DELETE `/api/trading/positions/{symbol}`
Close position
```bash
curl -X DELETE http://localhost:8000/api/trading/positions/BTC
```

#### POST `/api/trading/mode`
Switch trading mode
```bash
curl -X POST "http://localhost:8000/api/trading/mode?mode=live"
```

#### GET `/api/trading/portfolio`
Get portfolio summary
```bash
curl http://localhost:8000/api/trading/portfolio
```

### Performance Analytics

#### GET `/api/performance/metrics`
Get comprehensive metrics
```bash
curl http://localhost:8000/api/performance/metrics
```

Response includes:
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- And more...

#### GET `/api/performance/equity-curve`
Get equity curve data
```bash
curl "http://localhost:8000/api/performance/equity-curve?days=30"
```

#### GET `/api/performance/trades`
Get trade history
```bash
curl "http://localhost:8000/api/performance/trades?limit=100"
```

#### GET `/api/performance/drawdown`
Get drawdown analysis
```bash
curl http://localhost:8000/api/performance/drawdown
```

#### GET `/api/performance/returns-distribution`
Get distribution of returns
```bash
curl "http://localhost:8000/api/performance/returns-distribution?bins=20"
```

#### GET `/api/performance/monthly-returns`
Get monthly returns
```bash
curl http://localhost:8000/api/performance/monthly-returns
```

#### GET `/api/performance/risk-metrics`
Get risk metrics
```bash
curl http://localhost:8000/api/performance/risk-metrics
```

#### GET `/api/performance/period-comparison`
Compare periods
```bash
curl http://localhost:8000/api/performance/period-comparison
```

#### GET `/api/performance/attribution`
Get performance attribution
```bash
curl http://localhost:8000/api/performance/attribution
```

### Model Management

#### GET `/api/models/list`
List available models
```bash
curl http://localhost:8000/api/models/list
```

#### POST `/api/models/activate/{model_name}`
Activate a model
```bash
curl -X POST http://localhost:8000/api/models/activate/ensemble
```

**Models:**
- `lstm` - Bidirectional LSTM (602K parameters)
- `cnn` - Convolutional Neural Network (130K parameters)
- `attention` - Transformer Attention (410K parameters)
- `ensemble` - Voting Ensemble (1.1M parameters)

#### GET `/api/models/active`
Get active model info
```bash
curl http://localhost:8000/api/models/active
```

#### GET `/api/models/predictions`
Get recent predictions
```bash
curl "http://localhost:8000/api/models/predictions?limit=50"
```

#### POST `/api/models/train`
Start model training
```bash
curl -X POST "http://localhost:8000/api/models/train?model_name=ensemble"
```

#### GET `/api/models/performance-comparison`
Compare model performance
```bash
curl http://localhost:8000/api/models/performance-comparison
```

#### GET `/api/models/feature-importance`
Get feature importance
```bash
curl http://localhost:8000/api/models/feature-importance
```

#### GET `/api/models/confusion-matrix`
Get confusion matrix
```bash
curl http://localhost:8000/api/models/confusion-matrix
```

#### GET `/api/models/roc-curve`
Get ROC curve data
```bash
curl http://localhost:8000/api/models/roc-curve
```

#### GET `/api/models/training-history`
Get training history
```bash
curl http://localhost:8000/api/models/training-history
```

### Configuration & Settings

#### GET `/api/config/trading`
Get trading configuration
```bash
curl http://localhost:8000/api/config/trading
```

#### POST `/api/config/trading`
Update trading configuration
```bash
curl -X POST http://localhost:8000/api/config/trading \
  -H "Content-Type: application/json" \
  -d '{"max_positions": 25, "stop_loss_percent": 0.03}'
```

#### GET `/api/config/risk-limits`
Get risk limits
```bash
curl http://localhost:8000/api/config/risk-limits
```

#### POST `/api/config/risk-limits`
Update risk limits
```bash
curl -X POST http://localhost:8000/api/config/risk-limits \
  -H "Content-Type: application/json" \
  -d '{"max_drawdown_percent": 0.20}'
```

#### GET `/api/config/watchlist`
Get symbol watchlist
```bash
curl http://localhost:8000/api/config/watchlist
```

#### POST `/api/config/watchlist`
Add to watchlist
```bash
curl -X POST "http://localhost:8000/api/config/watchlist?symbol=XRP"
```

#### DELETE `/api/config/watchlist/{symbol}`
Remove from watchlist
```bash
curl -X DELETE http://localhost:8000/api/config/watchlist/XRP
```

#### GET `/api/config/api-status`
Get API connection status
```bash
curl http://localhost:8000/api/config/api-status
```

#### POST `/api/config/api-reconnect/{api_name}`
Reconnect to API
```bash
curl -X POST http://localhost:8000/api/config/api-reconnect/polygon_io
```

#### GET `/api/config/data-sources`
Get data sources
```bash
curl http://localhost:8000/api/config/data-sources
```

#### POST `/api/config/data-sources`
Update data source
```bash
curl -X POST "http://localhost:8000/api/config/data-sources?source_name=polygon.io&is_primary=true"
```

#### GET `/api/config/system-info`
Get system information
```bash
curl http://localhost:8000/api/config/system-info
```

#### POST `/api/config/password-change`
Change password
```bash
curl -X POST http://localhost:8000/api/config/password-change \
  -H "Content-Type: application/json" \
  -d '{"current_password": "admin", "new_password": "newpassword"}'
```

#### GET `/api/config/backup-config`
Backup configuration
```bash
curl http://localhost:8000/api/config/backup-config > config_backup.json
```

#### POST `/api/config/restore-config`
Restore configuration
```bash
curl -X POST http://localhost:8000/api/config/restore-config \
  -H "Content-Type: application/json" \
  -d @config_backup.json
```

## WebSocket Connection

Connect to the WebSocket endpoint for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/updates');

ws.onopen = () => {
  console.log('Connected');
  ws.send('Connected to trading dashboard');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected');
};
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid parameter value"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid password"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

## Authentication

Most endpoints are public. Protected endpoints require an Authorization header:

```bash
curl -H "Authorization: Bearer {token}" http://localhost:8000/api/endpoint
```

Get a token by logging in:
```bash
curl -X POST "http://localhost:8000/auth/login?password=admin"
```

## Rate Limiting

No rate limiting is currently implemented. In production, add:
- Rate limiting per IP
- Rate limiting per user
- Burst allowances

## CORS

CORS is enabled for:
- `http://localhost:3000` (development frontend)
- `http://localhost:8000` (development API)
- `https://defi-neural-network.vercel.app` (production)

## Performance

- **Typical Response Time**: 50-100ms
- **WebSocket Latency**: <50ms
- **Concurrent Connections**: Tested up to 100+

## Testing

### Using curl
```bash
# Test health
curl http://localhost:8000/health

# Test trading endpoint
curl http://localhost:8000/api/trading/status

# Test with POST
curl -X POST "http://localhost:8000/api/trading/start?mode=paper"
```

### Using Python
```python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Start trading
response = requests.post('http://localhost:8000/api/trading/start?mode=paper')
print(response.json())
```

### Using JavaScript
```javascript
// Fetch API
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(data => console.log(data));

// Start trading
fetch('http://localhost:8000/api/trading/start?mode=paper', {
  method: 'POST'
})
  .then(r => r.json())
  .then(data => console.log(data));
```

## Production Deployment

For production deployment:

1. Use environment variables for all secrets
2. Enable HTTPS/SSL
3. Add rate limiting
4. Add request validation
5. Use database instead of in-memory state
6. Add logging and monitoring
7. Setup health checks
8. Configure auto-restart
9. Add backup and recovery
10. Enable request/response compression

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### CORS Errors
Check that frontend URL is in CORS_ORIGINS environment variable

### WebSocket Connection Failed
Ensure WebSocket support is enabled and firewall allows connections

## Support

For issues or questions, refer to the project documentation or create an issue on GitHub.
