'use client'

import { useStartEngine, useStopEngine } from '@/hooks/useApi'
import { Play, Stop } from 'lucide-react'
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

  const handleStart = async () => {
    await startEngine.mutateAsync('paper')
  }

  const handleStop = async () => {
    await stopEngine.mutateAsync()
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

  return (
    <div className="dashboard-card">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold mb-2">Trading Engine</h2>
          <p className="text-sm text-muted-foreground">
            Status: <span className={status?.running ? 'text-green-500' : 'text-yellow-500'}>
              {status?.running ? 'Running' : 'Stopped'}
            </span>
          </p>
          <p className="text-sm text-muted-foreground">
            Mode: <span className="font-medium capitalize">{status?.trading_mode || 'paper'}</span>
          </p>
        </div>

        <div className="flex gap-3">
          <button
            onClick={handleStart}
            disabled={status?.running || startEngine.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            <Play className="w-4 h-4" />
            {startEngine.isPending ? 'Starting...' : 'Start'}
          </button>

          <button
            onClick={handleStop}
            disabled={!status?.running || stopEngine.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            <Stop className="w-4 h-4" />
            {stopEngine.isPending ? 'Stopping...' : 'Stop'}
          </button>
        </div>
      </div>
    </div>
  )
}
