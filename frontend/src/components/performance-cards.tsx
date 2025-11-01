'use client'

import type { PerformanceMetrics } from '@/types/api'
import { TrendingUp, TrendingDown } from 'lucide-react'

interface PerformanceCardsProps {
  metrics: PerformanceMetrics | undefined
  isLoading: boolean
}

export default function PerformanceCards({
  metrics,
  isLoading,
}: PerformanceCardsProps) {
  if (isLoading) {
    return (
      <div className="dashboard-grid">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="dashboard-card animate-pulse">
            <div className="h-4 bg-muted rounded w-1/2 mb-2"></div>
            <div className="h-6 bg-muted rounded w-3/4"></div>
          </div>
        ))}
      </div>
    )
  }

  const cards = metrics ? [
    {
      label: 'Sharpe Ratio',
      value: metrics.sharpe_ratio.toFixed(2),
      isGood: metrics.sharpe_ratio > 1.5,
    },
    {
      label: 'Total Return',
      value: (metrics.total_return * 100).toFixed(2) + '%',
      isGood: metrics.total_return > 0.05,
    },
    {
      label: 'Win Rate',
      value: (metrics.win_rate * 100).toFixed(1) + '%',
      isGood: metrics.win_rate > 0.55,
    },
    {
      label: 'Max Drawdown',
      value: (metrics.max_drawdown * 100).toFixed(2) + '%',
      isGood: metrics.max_drawdown > -0.2,
    },
  ] : []

  return (
    <div className="dashboard-grid">
      {cards.map((card) => (
        <div key={card.label} className="dashboard-card">
          <p className="text-sm text-muted-foreground mb-1">{card.label}</p>
          <div className="flex items-center justify-between">
            <span className="text-2xl font-bold">{card.value}</span>
            {card.isGood ? (
              <TrendingUp className="w-5 h-5 text-green-500" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-500" />
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
