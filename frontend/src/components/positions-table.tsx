'use client'

import { useState, useEffect } from 'react'
import { useWebSocket } from '@/hooks/useWebSocket'
import { TrendingUp, TrendingDown } from 'lucide-react'
import type { Position } from '@/types/api'

interface PositionsTableProps {
  positions: Position[]
  isLoading: boolean
}

interface PriceData {
  [symbol: string]: {
    current: number
    change: number
    changePercent: number
    updatedAt: number
  }
}

export default function PositionsTable({
  positions,
  isLoading,
}: PositionsTableProps) {
  const [prices, setPrices] = useState<PriceData>({})

  // Listen for real-time price updates
  useWebSocket({
    onPriceUpdate: (priceData) => {
      setPrices((prev) => ({
        ...prev,
        [priceData.symbol]: {
          current: priceData.price,
          change: priceData.change,
          changePercent: (priceData.change / priceData.price) * 100,
          updatedAt: Date.now(),
        },
      }))
    },
  })

  if (isLoading) {
    return (
      <div className="animate-pulse space-y-2">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="h-12 bg-muted rounded"></div>
        ))}
      </div>
    )
  }

  if (positions.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No open positions
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-3 px-4 font-semibold">Symbol</th>
            <th className="text-right py-3 px-4 font-semibold">Qty</th>
            <th className="text-right py-3 px-4 font-semibold">Entry Price</th>
            <th className="text-right py-3 px-4 font-semibold">Current Price</th>
            <th className="text-right py-3 px-4 font-semibold">Change</th>
            <th className="text-right py-3 px-4 font-semibold">Entry Time</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((position) => {
            const priceInfo = prices[position.symbol]
            const isPositiveChange = priceInfo?.change >= 0
            const pnl = (priceInfo?.current ?? position.entry_price - position.entry_price) * position.quantity

            return (
              <tr
                key={position.symbol}
                className="border-b border-border hover:bg-muted/50 transition-colors"
              >
                <td className="py-3 px-4 font-medium">{position.symbol}</td>
                <td className="text-right py-3 px-4">{position.quantity.toFixed(4)}</td>
                <td className="text-right py-3 px-4">${position.entry_price.toFixed(2)}</td>
                <td className="text-right py-3 px-4 font-medium">
                  {priceInfo?.current ? (
                    <span>${priceInfo.current.toFixed(2)}</span>
                  ) : (
                    <span className="text-muted-foreground">--</span>
                  )}
                </td>
                <td className="text-right py-3 px-4">
                  {priceInfo ? (
                    <div className="flex items-center justify-end gap-1">
                      {isPositiveChange ? (
                        <TrendingUp className="w-4 h-4 text-accent" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-destructive" />
                      )}
                      <span
                        className={
                          isPositiveChange ? 'text-accent' : 'text-destructive'
                        }
                      >
                        {isPositiveChange ? '+' : ''}
                        {priceInfo.changePercent.toFixed(2)}%
                      </span>
                    </div>
                  ) : (
                    <span className="text-muted-foreground">--</span>
                  )}
                </td>
                <td className="text-right py-3 px-4 text-xs text-muted-foreground">
                  {position.entry_time.slice(0, 10)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
