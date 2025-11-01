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
├── config/                  # Configuration & API setup
│   ├── api_config.py       # API credentials and endpoints
│   ├── model_config.py     # Neural network parameters
│   └── constants.py        # System-wide constants
├── data/                   # Data pipeline & APIs
├── features/               # Feature engineering
├── models/                 # Neural network architectures
├── training/               # Training pipelines
├── evaluation/             # Backtesting & metrics
├── agents/                 # Pydantic AI agents
├── inference/              # Real-time prediction
├── integrations/           # Third-party integrations
├── cli/                    # Command-line interface
├── tests/                  # Unit & integration tests
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

## Next Steps

1. **Review** `FINANCIAL_NN_COMPLETE_GUIDE.md` for detailed architecture
2. **Start Stage 1**: Initialize core modules
3. **Build Data Pipeline**: Connect to Polygon.io
4. **Train Models**: LSTM, Attention, CNN
5. **Deploy**: Production inference engine

## Development Workflow

### Day 1: Setup & Data (Stages 1-2)
- Initialize project & dependencies
- Connect to data APIs
- Validate data quality

### Day 2: Features & Models (Stages 3-4)
- Compute technical indicators
- Build neural network architectures
- Test forward passes

### Days 3+: Training, Backtesting & Production
- Train models with walk-forward validation
- Integrate AI agents
- Deploy real-time inference

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
