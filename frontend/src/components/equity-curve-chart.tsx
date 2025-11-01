'use client'

import { useEquityCurve } from '@/hooks/useApi'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

export default function EquityCurveChart() {
  const { data: equityCurve, isLoading } = useEquityCurve(30)

  if (isLoading) {
    return (
      <div className="animate-pulse h-72 bg-muted rounded"></div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={equityCurve || []}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
        <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" />
        <YAxis stroke="hsl(var(--muted-foreground))" />
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '0.5rem',
          }}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke="hsl(var(--accent))"
          dot={false}
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
