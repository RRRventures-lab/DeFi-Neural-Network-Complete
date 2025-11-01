'use client'

import type { Position } from '@/types/api'

interface PositionsTableProps {
  positions: Position[]
  isLoading: boolean
}

export default function PositionsTable({
  positions,
  isLoading,
}: PositionsTableProps) {
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
            <th className="text-right py-3 px-4 font-semibold">Entry</th>
            <th className="text-right py-3 px-4 font-semibold">Time</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((position) => (
            <tr key={position.symbol} className="border-b border-border hover:bg-muted/50">
              <td className="py-3 px-4 font-medium">{position.symbol}</td>
              <td className="text-right py-3 px-4">{position.quantity}</td>
              <td className="text-right py-3 px-4">${position.entry_price}</td>
              <td className="text-right py-3 px-4 text-xs text-muted-foreground">
                {position.entry_time.slice(0, 10)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
