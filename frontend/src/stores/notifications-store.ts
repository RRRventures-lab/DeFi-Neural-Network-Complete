import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

export interface Notification {
  id: string
  title: string
  message: string
  severity: 'info' | 'warning' | 'error' | 'success'
  timestamp: number
  duration?: number // milliseconds, undefined = persistent
}

interface NotificationsStore {
  notifications: Notification[]
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void
  removeNotification: (id: string) => void
  clearNotifications: () => void
}

export const useNotificationsStore = create<NotificationsStore>()(
  devtools((set) => ({
    notifications: [],
    addNotification: (notification) => {
      const id = `${Date.now()}-${Math.random()}`
      const newNotification: Notification = {
        ...notification,
        id,
        timestamp: Date.now(),
      }

      set((state) => ({
        notifications: [newNotification, ...state.notifications],
      }))

      // Auto-remove if duration is specified
      if (notification.duration) {
        setTimeout(() => {
          set((state) => ({
            notifications: state.notifications.filter((n) => n.id !== id),
          }))
        }, notification.duration)
      }
    },
    removeNotification: (id) => {
      set((state) => ({
        notifications: state.notifications.filter((n) => n.id !== id),
      }))
    },
    clearNotifications: () => {
      set({ notifications: [] })
    },
  })),
)
