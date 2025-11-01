'use client'

import { AlertCircle, Check, Info, X, AlertTriangle } from 'lucide-react'
import { useNotificationsStore } from '@/stores/notifications-store'

export default function NotificationsContainer() {
  const notifications = useNotificationsStore((state) => state.notifications)
  const removeNotification = useNotificationsStore((state) => state.removeNotification)

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-md">
      {notifications.map((notification) => {
        const icons = {
          info: <Info className="w-5 h-5" />,
          success: <Check className="w-5 h-5" />,
          warning: <AlertTriangle className="w-5 h-5" />,
          error: <AlertCircle className="w-5 h-5" />,
        }

        const bgColors = {
          info: 'bg-blue-500/10 border-blue-500/30',
          success: 'bg-accent/10 border-accent/30',
          warning: 'bg-yellow-500/10 border-yellow-500/30',
          error: 'bg-destructive/10 border-destructive/30',
        }

        const textColors = {
          info: 'text-blue-500',
          success: 'text-accent',
          warning: 'text-yellow-500',
          error: 'text-destructive',
        }

        const iconBgColors = {
          info: 'bg-blue-500/20',
          success: 'bg-accent/20',
          warning: 'bg-yellow-500/20',
          error: 'bg-destructive/20',
        }

        return (
          <div
            key={notification.id}
            className={`rounded-lg border p-4 flex gap-3 animate-in slide-in-from-right-full ${bgColors[notification.severity]}`}
          >
            <div className={`p-2 rounded-lg ${iconBgColors[notification.severity]} flex-shrink-0`}>
              <div className={textColors[notification.severity]}>
                {icons[notification.severity]}
              </div>
            </div>

            <div className="flex-1">
              <h3 className={`font-semibold ${textColors[notification.severity]}`}>
                {notification.title}
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                {notification.message}
              </p>
            </div>

            <button
              onClick={() => removeNotification(notification.id)}
              className="flex-shrink-0 text-muted-foreground hover:text-foreground transition"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        )
      })}
    </div>
  )
}
