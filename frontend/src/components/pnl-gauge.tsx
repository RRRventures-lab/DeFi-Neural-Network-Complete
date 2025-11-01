'use client'

import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { useWebSocket } from '@/hooks/useWebSocket'

interface PnLGaugeProps {
  initialPnL?: number
  initialChange?: number
  initialChangePercent?: number
}

export default function PnLGauge({
  initialPnL = 0,
  initialChange = 0,
  initialChangePercent = 0,
}: PnLGaugeProps) {
  const [pnl, setPnL] = useState(initialPnL)
  const [change, setChange] = useState(initialChange)
  const [changePercent, setChangePercent] = useState(initialChangePercent)
  const [isPositive, setIsPositive] = useState(pnl >= 0)

  // Listen to real-time P&L updates
  useWebSocket({
    onPnlUpdate: (pnlData) => {
      setPnL(pnlData.current)
      setChange(pnlData.change)
      setChangePercent(pnlData.change_percent)
      setIsPositive(pnlData.current >= 0)
    },
  })

  // Calculate gauge rotation based on P&L
  const maxPnL = 10000
  const minPnL = -10000
  const clampedPnL = Math.max(minPnL, Math.min(maxPnL, pnl))
  const gaugePercent = ((clampedPnL - minPnL) / (maxPnL - minPnL)) * 100
  const gaugeDegrees = 180 * (gaugePercent / 100) - 90 // -90 to 90 degrees

  return (
    <div className="flex flex-col items-center justify-center p-8 bg-card rounded-lg border">
      <h3 className="text-lg font-semibold text-foreground mb-6">P&L Gauge</h3>

      {/* Gauge Container */}
      <div className="relative w-64 h-40 mb-8">
        {/* Gauge Background Arc */}
        <svg
          viewBox="0 0 200 120"
          className="w-full h-full"
          style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' }}
        >
          {/* Loss side (red) */}
          <path
            d="M 20 100 A 80 80 0 0 1 50 30"
            fill="none"
            stroke="hsl(var(--destructive))"
            strokeWidth="8"
            opacity="0.3"
          />

          {/* Neutral zone (gray) */}
          <path
            d="M 50 30 A 80 80 0 0 1 150 30"
            fill="none"
            stroke="hsl(var(--muted-foreground))"
            strokeWidth="8"
            opacity="0.2"
          />

          {/* Profit side (green) */}
          <path
            d="M 150 30 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="hsl(var(--accent))"
            strokeWidth="8"
            opacity="0.3"
          />

          {/* Needle */}
          <g transform={`translate(100, 100) rotate(${gaugeDegrees})`}>
            <line
              x1="0"
              y1="0"
              x2="0"
              y2="-70"
              stroke={isPositive ? 'hsl(var(--accent))' : 'hsl(var(--destructive))'}
              strokeWidth="3"
              strokeLinecap="round"
            />
            <circle
              cx="0"
              cy="0"
              r="6"
              fill={isPositive ? 'hsl(var(--accent))' : 'hsl(var(--destructive))'}
            />
          </g>

          {/* Labels */}
          <text
            x="20"
            y="115"
            fontSize="12"
            fill="hsl(var(--muted-foreground))"
            textAnchor="middle"
          >
            -$10K
          </text>
          <text
            x="100"
            y="115"
            fontSize="12"
            fill="hsl(var(--muted-foreground))"
            textAnchor="middle"
            fontWeight="bold"
          >
            $0
          </text>
          <text
            x="180"
            y="115"
            fontSize="12"
            fill="hsl(var(--muted-foreground))"
            textAnchor="middle"
          >
            +$10K
          </text>
        </svg>
      </div>

      {/* P&L Value Display */}
      <div className="text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <span className="text-4xl font-bold text-foreground">
            ${Math.abs(pnl).toFixed(2)}
          </span>
          {isPositive ? (
            <TrendingUp className="w-8 h-8 text-accent" />
          ) : (
            <TrendingDown className="w-8 h-8 text-destructive" />
          )}
        </div>

        {/* Change Display */}
        <div className="flex items-center justify-center gap-1">
          <span
            className={`text-lg font-semibold ${
              isPositive ? 'text-accent' : 'text-destructive'
            }`}
          >
            {isPositive ? '+' : ''}${change.toFixed(2)}
          </span>
          <span
            className={`text-sm ${isPositive ? 'text-accent' : 'text-destructive'}`}
          >
            ({changePercent.toFixed(2)}%)
          </span>
        </div>
      </div>

      {/* Status Indicator */}
      <div className="mt-6 flex items-center gap-2 px-4 py-2 bg-muted rounded">
        <div
          className={`w-2 h-2 rounded-full ${
            isPositive ? 'bg-accent' : 'bg-destructive'
          }`}
        />
        <span className="text-sm text-muted-foreground">
          {isPositive ? 'Profitable' : 'Loss'} Position
        </span>
      </div>
    </div>
  )
}
