# Defi-Neural-Network

A production-grade financial neural network system combining institutional data APIs, advanced machine learning, and Claude 4 AI reasoning for DeFi and traditional markets.

## Quick Start

### 1. Setup Environment

```bash
cd Defi-Neural-Network
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

The `.env` file already contains your API keys:
- **Polygon.io**: For stock & ETF data
- **Coinbase**: For cryptocurrency data
- **Perplexity**: For AI-powered research
- **Deepseek**: For alternative AI analysis

### 3. Run Tests

```bash
python -m pytest tests/
```

## Project Structure

```
Defi-Neural-Network/
├── trading_engine/                  # Integrated trading system (Stage 9)
│   ├── trading_engine.py           # Core engine orchestration
│   ├── execution_manager.py        # Order execution & slippage
│   ├── performance_monitor.py      # Metrics & analytics
│   ├── deployment_manager.py       # Live trading support
│   └── trading_agent.py            # Autonomous decision maker
├── multi_asset/                     # Multi-asset trading (Stage 8)
├── advanced/                        # Advanced features (Stage 7)
├── training/                        # Training pipelines (Stage 4)
├── evaluation/                      # Backtesting & metrics (Stage 5)
├── risk_management/                 # Risk management (Stage 6)
├── models/                          # Neural networks (Stage 3)
├── features/                        # Feature engineering (Stage 2)
├── data/                            # Data pipeline (Stage 1)
├── config/                          # Configuration & setup
├── agents/                          # Pydantic AI agents
├── integrations/                    # Third-party integrations
├── cli/                             # Command-line interface
├── backend/                         # FastAPI backend (Phase 1)
│   ├── main.py                     # FastAPI application
│   ├── routers/                    # API endpoint routers
│   ├── core/                       # Module wrappers
│   ├── requirements.txt            # Dependencies
│   └── API_DOCUMENTATION.md        # Complete API reference
├── tests/                           # Unit & integration tests
├── PHASE_1_BACKEND_COMPLETE.md     # Phase 1 summary
└── FINANCIAL_NN_COMPLETE_GUIDE.md  # Detailed implementation guide
```

## Key Features

- **Multi-Source Data**: Polygon.io, Coinbase, Massive.com
- **Advanced ML**: LSTM, Attention, CNN, Ensemble models
- **Real-Time Inference**: <100ms prediction latency
- **Backtesting Engine**: Walk-forward validation, performance analysis
- **Pydantic AI Agents**: Structured, type-safe automation
- **10+ Integrations**: Sentiment, options, economic data, more
- **Production Ready**: Monitoring, logging, error handling

## API Keys Used

| Service | Purpose |
|---------|---------|
| Polygon.io | Stock quotes, historical data, technical indicators |
| Coinbase | Cryptocurrency market data, wallet integration |
| Perplexity | AI-powered market research |
| Deepseek | Alternative LLM reasoning |

## Development Status

### ✅ Completed
- **Stages 1-9**: Full DeFi Neural Network trading system (2,800+ lines)
  - Stage 1: Data pipeline with async API integration
  - Stage 2: 34+ technical indicators
  - Stage 3: LSTM, CNN, Attention, Ensemble neural networks
  - Stage 4: Walk-forward backtesting
  - Stage 5: Trading logic implementation
  - Stage 6: Risk management system
  - Stage 7: Advanced features (tax, scenarios, options)
  - Stage 8: Multi-asset trading (crypto, forex, derivatives)
  - Stage 9: Integrated trading engine with autonomous agent

- **Phase 1**: FastAPI Backend (1,200+ lines)
  - 43 REST/WebSocket endpoints
  - Trading operations, performance analytics, model management
  - Real-time WebSocket support
  - Production-ready deployment

### 🚀 Next Steps: Phase 2 - Frontend Development
1. **Generate** dashboard layouts with Vercel v0
2. **Create** Next.js 14 project
3. **Implement** API client and WebSocket integration
4. **Build** real-time dashboard UI
5. **Deploy** to Vercel

## Getting Started with Backend

### Quick Start

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env

# 3. Start server
python3 -m uvicorn main:app --port 8000 --reload
# OR use the startup script
./start.sh

# 4. Access API
# Swagger UI: http://localhost:8000/docs
# API: http://localhost:8000
# WebSocket: ws://localhost:8000/ws/updates
```

### API Endpoints

- **Trading**: Start/stop engine, process signals, manage orders
- **Performance**: Metrics, equity curve, trade analysis, drawdowns
- **Models**: List models, make predictions, compare performance
- **Configuration**: Trading settings, risk limits, watchlist
- **WebSocket**: Real-time updates for live dashboard

See `backend/API_DOCUMENTATION.md` for complete endpoint reference.

## Development Workflow

### Completed: Stages 1-9 (Trading Engine)
- Data pipeline & feature engineering
- Neural network models
- Backtesting framework
- Risk management
- Multi-asset trading
- Integrated trading engine

### In Progress: Phase 1 (Backend) ✅
- FastAPI server with 43 endpoints
- Real-time WebSocket support
- Production deployment ready

### Next: Phase 2-7 (Frontend & Dashboard)
- Next.js dashboard with Vercel v0
- Real-time trading interface
- Performance analytics
- Model comparison UI
- Settings and configuration

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Sharpe Ratio | > 1.5 | ⏳ |
| Win Rate | > 55% | ⏳ |
| Max Drawdown | < -20% | ⏳ |
| Inference Latency | < 100ms | ⏳ |

## Installation Issues?

If you encounter issues with PyTorch/TensorFlow:

```bash
# For CPU-only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon:
pip install torch::*=*=cpu
```

## License

Proprietary - All Rights Reserved

## Support

For issues, check the debugging section in `FINANCIAL_NN_COMPLETE_GUIDE.md` or review error logs in `financial_nn.log`.
