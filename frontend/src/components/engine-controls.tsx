'use client'

import { useState, useEffect } from 'react'
import { useStartEngine, useStopEngine } from '@/hooks/useApi'
import { useWebSocket } from '@/hooks/useWebSocket'
import { Play, Stop, Zap, AlertCircle } from 'lucide-react'
import type { EngineStatus } from '@/types/api'

interface EngineControlsProps {
  status: EngineStatus | undefined
  isLoading: boolean
}

export default function EngineControls({
  status,
  isLoading,
}: EngineControlsProps) {
  const startEngine = useStartEngine()
  const stopEngine = useStopEngine()
  const [localStatus, setLocalStatus] = useState<EngineStatus | undefined>(status)
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now())
  const [uptime, setUptime] = useState<string>('')

  // Listen to real-time status updates via WebSocket
  useWebSocket({
    onMessage: (data) => {
      if (data.type === 'status_update') {
        setLocalStatus(data.data)
        setLastUpdate(Date.now())
      }
    },
  })

  // Update uptime display
  useEffect(() => {
    if (!localStatus?.running) return

    const interval = setInterval(() => {
      const secondsRunning = Math.floor((Date.now() - lastUpdate) / 1000)
      const hours = Math.floor(secondsRunning / 3600)
      const minutes = Math.floor((secondsRunning % 3600) / 60)
      const seconds = secondsRunning % 60

      if (hours > 0) {
        setUptime(`${hours}h ${minutes}m`)
      } else if (minutes > 0) {
        setUptime(`${minutes}m ${seconds}s`)
      } else {
        setUptime(`${seconds}s`)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [localStatus?.running, lastUpdate])

  // Use local status if available, fall back to prop
  const displayStatus = localStatus || status

  const handleStart = async () => {
    try {
      await startEngine.mutateAsync('paper')
      setLocalStatus((prev) => prev ? { ...prev, running: true } : undefined)
    } catch (error) {
      console.error('Failed to start engine:', error)
    }
  }

  const handleStop = async () => {
    try {
      await stopEngine.mutateAsync()
      setLocalStatus((prev) => prev ? { ...prev, running: false } : undefined)
    } catch (error) {
      console.error('Failed to stop engine:', error)
    }
  }

  if (isLoading) {
    return (
      <div className="dashboard-card">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-muted rounded w-1/4"></div>
          <div className="h-10 bg-muted rounded w-full"></div>
        </div>
      </div>
    )
  }

  const isRunning = displayStatus?.running
  const statusColor = isRunning ? 'text-accent' : 'text-muted-foreground'
  const statusBgColor = isRunning ? 'bg-accent/10' : 'bg-muted/50'

  return (
    <div className={`dashboard-card border-2 transition-colors ${isRunning ? 'border-accent/30' : 'border-border'}`}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-4">
            <div className={`p-2 rounded-lg ${statusBgColor}`}>
              <Zap className={`w-5 h-5 ${statusColor}`} />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Trading Engine</h2>
              <p className="text-xs text-muted-foreground">Real-time Status</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-muted-foreground mb-1">Status</p>
              <div className="flex items-center gap-2">
                <div
                  className={`w-2.5 h-2.5 rounded-full animate-pulse ${
                    isRunning ? 'bg-accent' : 'bg-muted-foreground'
                  }`}
                />
                <span className={`font-medium ${statusColor}`}>
                  {isRunning ? 'Running' : 'Stopped'}
                </span>
              </div>
            </div>

            <div>
              <p className="text-xs text-muted-foreground mb-1">Mode</p>
              <span className="font-medium capitalize">
                {displayStatus?.trading_mode || 'paper'}
              </span>
            </div>

            {isRunning && (
              <>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Uptime</p>
                  <span className="font-mono text-sm">{uptime}</span>
                </div>

                <div>
                  <p className="text-xs text-muted-foreground mb-1">Capital</p>
                  <span className="font-medium">
                    ${displayStatus?.capital?.toFixed(2) || '0.00'}
                  </span>
                </div>
              </>
            )}

            <div>
              <p className="text-xs text-muted-foreground mb-1">Positions</p>
              <span className="font-medium">
                {displayStatus?.positions?.length || 0}
              </span>
            </div>

            {isRunning && displayStatus?.pnl && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">P&L</p>
                <span
                  className={`font-medium ${
                    displayStatus.pnl >= 0 ? 'text-accent' : 'text-destructive'
                  }`}
                >
                  {displayStatus.pnl >= 0 ? '+' : ''}${displayStatus.pnl.toFixed(2)}
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <button
            onClick={handleStart}
            disabled={isRunning || startEngine.isPending}
            className="flex items-center gap-2 px-4 py-2.5 bg-accent hover:bg-accent/90 text-accent-foreground rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
          >
            <Play className="w-4 h-4" />
            {startEngine.isPending ? 'Starting...' : 'Start'}
          </button>

          <button
            onClick={handleStop}
            disabled={!isRunning || stopEngine.isPending}
            className="flex items-center gap-2 px-4 py-2.5 bg-destructive hover:bg-destructive/90 text-destructive-foreground rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
          >
            <Stop className="w-4 h-4" />
            {stopEngine.isPending ? 'Stopping...' : 'Stop'}
          </button>
        </div>
      </div>

      {startEngine.isError || stopEngine.isError ? (
        <div className="mt-4 p-3 bg-destructive/10 border border-destructive/30 rounded-lg flex gap-2">
          <AlertCircle className="w-4 h-4 text-destructive flex-shrink-0 mt-0.5" />
          <p className="text-sm text-destructive">
            {startEngine.error?.message || stopEngine.error?.message || 'An error occurred'}
          </p>
        </div>
      ) : null}
    </div>
  )
}
