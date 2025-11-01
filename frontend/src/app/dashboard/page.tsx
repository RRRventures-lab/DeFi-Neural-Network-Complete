'use client'

import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { useAuthStore } from '@/stores/auth-store'
import { useNotificationsStore } from '@/stores/notifications-store'
import { useEngineStatus, useMetrics, usePositions } from '@/hooks/useApi'
import { useWebSocket } from '@/hooks/useWebSocket'
import DashboardHeader from '@/components/dashboard-header'
import EngineControls from '@/components/engine-controls'
import PositionsTable from '@/components/positions-table'
import PerformanceCards from '@/components/performance-cards'
import EquityCurveChart from '@/components/equity-curve-chart'
import PnLGauge from '@/components/pnl-gauge'
import NotificationsContainer from '@/components/notifications-container'

export default function DashboardPage() {
  const router = useRouter()
  const { isAuthenticated, logout } = useAuthStore()
  const addNotification = useNotificationsStore((state) => state.addNotification)

  // API queries
  const { data: engineStatus, isLoading: engineLoading } = useEngineStatus()
  const { data: metrics, isLoading: metricsLoading } = useMetrics()
  const { data: positions, isLoading: positionsLoading } = usePositions()

  // WebSocket with event handlers
  const { isConnected } = useWebSocket({
    enabled: isAuthenticated,
    onConnect: () => {
      addNotification({
        title: 'Connected',
        message: 'Real-time connection established',
        severity: 'success',
        duration: 3000,
      })
    },
    onDisconnect: () => {
      addNotification({
        title: 'Disconnected',
        message: 'Lost real-time connection. Attempting to reconnect...',
        severity: 'warning',
      })
    },
    onError: (error) => {
      addNotification({
        title: 'Connection Error',
        message: 'Failed to connect to real-time updates',
        severity: 'error',
        duration: 5000,
      })
    },
    onAlert: (alert) => {
      addNotification({
        title: alert.title,
        message: alert.message,
        severity: alert.severity,
        duration: alert.severity === 'error' ? 0 : 5000,
      })
    },
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
        {/* Notifications */}
        <NotificationsContainer />

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

        {/* Charts and Gauge */}
        <div className="grid gap-6 md:grid-cols-3">
          <div className="dashboard-card md:col-span-2">
            <h2 className="text-lg font-semibold mb-4">Equity Curve</h2>
            <EquityCurveChart />
          </div>

          <div>
            <PnLGauge
              initialPnL={engineStatus?.pnl || 0}
              initialChange={0}
              initialChangePercent={0}
            />
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

        {/* Additional Trading Stats */}
        <div className="grid gap-6 md:grid-cols-2">
          <div className="dashboard-card">
            <h2 className="text-lg font-semibold mb-4">Trading Activity</h2>
            <div className="space-y-3">
              <div className="flex justify-between items-center py-2 border-b border-border">
                <span className="text-muted-foreground">Total Trades</span>
                <span className="font-medium">{engineStatus?.positions?.length || 0}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-border">
                <span className="text-muted-foreground">Capital Deployed</span>
                <span className="font-medium">${engineStatus?.capital?.toFixed(2) || '0.00'}</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-muted-foreground">Status</span>
                <span className={`font-medium ${engineStatus?.running ? 'text-accent' : 'text-muted-foreground'}`}>
                  {engineStatus?.running ? 'Live' : 'Paused'}
                </span>
              </div>
            </div>
          </div>

          <div className="dashboard-card">
            <h2 className="text-lg font-semibold mb-4">System Health</h2>
            <div className="space-y-3">
              <div className="flex justify-between items-center py-2 border-b border-border">
                <span className="text-muted-foreground">WebSocket</span>
                <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-accent animate-pulse' : 'bg-muted-foreground'}`} />
              </div>
              <div className="flex justify-between items-center py-2 border-b border-border">
                <span className="text-muted-foreground">Trading Mode</span>
                <span className="font-medium capitalize">{engineStatus?.trading_mode || 'paper'}</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-muted-foreground">Last Update</span>
                <span className="text-xs text-muted-foreground">Just now</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
