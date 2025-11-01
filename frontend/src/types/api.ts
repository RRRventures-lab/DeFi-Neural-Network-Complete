// Trading Engine Types
export interface EngineStatus {
  running: boolean;
  trading_mode: 'paper' | 'live';
  initial_capital: number;
  current_value: number;
  total_pnl: number;
  pnl_percent: number;
  open_positions_count: number;
  signals_processed: number;
  orders_executed: number;
  timestamp: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  entry_time: string;
}

export interface Order {
  order_id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  type: 'market' | 'limit' | 'stop';
  executed_at: string;
}

// Performance Types
export interface PerformanceMetrics {
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  total_return: number;
  annual_return: number;
  win_rate: number;
  profit_factor: number;
  consecutive_wins: number;
  consecutive_losses: number;
  avg_win_size: number;
  avg_loss_size: number;
  best_trade: number;
  worst_trade: number;
  timestamp: string;
}

export interface EquityCurvePoint {
  date: string;
  value: number;
  daily_return: number;
}

export interface Trade {
  trade_id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  entry_price: number;
  entry_time: string;
  exit_price: number;
  exit_time: string;
  pnl: number;
  pnl_percent: number;
  duration_hours: number;
}

// Model Types
export interface Model {
  name: string;
  type: string;
  parameters: number;
  status: string;
  accuracy: number;
  precision: number;
  recall: number;
}

export interface Prediction {
  prediction_id: string;
  symbol: string;
  timestamp: string;
  prediction: 'buy' | 'hold' | 'sell';
  confidence: number;
  probability_buy: number;
  probability_hold: number;
  probability_sell: number;
  actual: 'buy' | 'hold' | 'sell';
  correct: boolean;
}

// Configuration Types
export interface TradingConfig {
  initial_capital: number;
  max_position_size: number;
  max_positions: number;
  max_leverage: number;
  use_stop_loss: boolean;
  stop_loss_percent: number;
  use_take_profit: boolean;
  take_profit_percent: number;
  rebalance_frequency: string;
  enable_shorting: boolean;
}

export interface RiskLimits {
  max_drawdown_percent: number;
  max_daily_loss_percent: number;
  max_position_concentration: number;
  min_position_size: number;
  var_confidence_level: number;
  max_sector_allocation: number;
}

// Response Types
export interface ApiResponse<T> {
  data: T;
  timestamp: string;
  status: 'success' | 'error';
}

export interface ApiError {
  detail: string;
  status_code: number;
}
