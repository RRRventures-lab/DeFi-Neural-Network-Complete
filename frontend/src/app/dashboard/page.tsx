'use client'

import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { useAuthStore } from '@/stores/auth-store'
import { useEngineStatus, useMetrics, usePositions } from '@/hooks/useApi'
import { useWebSocket } from '@/hooks/useWebSocket'
import DashboardHeader from '@/components/dashboard-header'
import EngineControls from '@/components/engine-controls'
import PositionsTable from '@/components/positions-table'
import PerformanceCards from '@/components/performance-cards'
import EquityCurveChart from '@/components/equity-curve-chart'

export default function DashboardPage() {
  const router = useRouter()
  const { isAuthenticated, logout } = useAuthStore()
  
  // API queries
  const { data: engineStatus, isLoading: engineLoading } = useEngineStatus()
  const { data: metrics, isLoading: metricsLoading } = useMetrics()
  const { data: positions, isLoading: positionsLoading } = usePositions()

  // WebSocket
  const { isConnected } = useWebSocket({
    enabled: isAuthenticated,
    onMessage: (data) => {
      console.log('WebSocket message:', data)
      // Update stores with real-time data
    },
  })

  // Auth check
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login')
    }
  }, [isAuthenticated, router])

  if (!isAuthenticated) {
    return null
  }

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader onLogout={logout} wsConnected={isConnected} />

      <main className="container mx-auto py-6 space-y-6">
        {/* Engine Controls */}
        <EngineControls 
          status={engineStatus}
          isLoading={engineLoading}
        />

        {/* Performance Metrics */}
        <PerformanceCards 
          metrics={metrics}
          isLoading={metricsLoading}
        />

        {/* Charts */}
        <div className="grid gap-6 md:grid-cols-2">
          <div className="dashboard-card">
            <h2 className="text-lg font-semibold mb-4">Equity Curve</h2>
            <EquityCurveChart />
          </div>

          <div className="dashboard-card">
            <h2 className="text-lg font-semibold mb-4">Market Data</h2>
            <div className="h-72 flex items-center justify-center text-muted-foreground">
              Coming soon...
            </div>
          </div>
        </div>

        {/* Positions */}
        <div className="dashboard-card">
          <h2 className="text-lg font-semibold mb-4">Open Positions</h2>
          <PositionsTable 
            positions={positions || []}
            isLoading={positionsLoading}
          />
        </div>
      </main>
    </div>
  )
}
