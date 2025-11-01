'use client'

import { LogOut, Wifi, WifiOff } from 'lucide-react'

interface DashboardHeaderProps {
  onLogout: () => void
  wsConnected: boolean
}

export default function DashboardHeader({
  onLogout,
  wsConnected,
}: DashboardHeaderProps) {
  return (
    <header className="border-b border-border bg-card">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">Trading Dashboard</h1>
          <span className="text-sm text-muted-foreground">DeFi Neural Network</span>
        </div>

        <div className="flex items-center gap-4">
          {/* WebSocket Status */}
          <div className="flex items-center gap-2">
            {wsConnected ? (
              <>
                <Wifi className="w-4 h-4 text-green-500" />
                <span className="text-sm text-green-500">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-red-500" />
                <span className="text-sm text-red-500">Disconnected</span>
              </>
            )}
          </div>

          {/* Logout Button */}
          <button
            onClick={onLogout}
            className="flex items-center gap-2 px-4 py-2 text-sm rounded-lg bg-muted hover:bg-muted/80 transition"
          >
            <LogOut className="w-4 h-4" />
            Logout
          </button>
        </div>
      </div>
    </header>
  )
}
