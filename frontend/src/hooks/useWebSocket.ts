import { useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { Position, PerformanceMetrics } from '@/types/api';

export interface WebSocketPosition extends Position {
  updated_at?: string;
}

export interface WebSocketUpdate {
  type: 'position_update' | 'pnl_update' | 'price_update' | 'alert' | 'metrics_update';
  data: any;
  timestamp?: number;
}

export interface WebSocketAlert {
  id: string;
  title: string;
  message: string;
  severity: 'info' | 'warning' | 'error' | 'success';
  timestamp: number;
}

interface UseWebSocketOptions {
  enabled?: boolean;
  onPositionUpdate?: (position: WebSocketPosition) => void;
  onPnlUpdate?: (pnl: { current: number; change: number; change_percent: number }) => void;
  onPriceUpdate?: (price: { symbol: string; price: number; change: number }) => void;
  onAlert?: (alert: WebSocketAlert) => void;
  onMetricsUpdate?: (metrics: Partial<PerformanceMetrics>) => void;
  onMessage?: (data: WebSocketUpdate) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: any) => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    enabled = true,
    onPositionUpdate,
    onPnlUpdate,
    onPriceUpdate,
    onAlert,
    onMetricsUpdate,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const socketRef = useRef<Socket | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (!enabled || socketRef.current?.connected) return;

    try {
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/updates';

      // Extract base URL for socket.io
      const baseUrl = wsUrl.replace(/^ws/, 'http').replace(/\/ws\/updates$/, '');

      socketRef.current = io(baseUrl, {
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: maxReconnectAttempts,
        transports: ['websocket', 'polling'],
      });

      socketRef.current.on('connect', () => {
        console.log('WebSocket connected');
        reconnectAttempts.current = 0;
        onConnect?.();
      });

      // Handle generic message
      socketRef.current.on('message', (data: WebSocketUpdate) => {
        onMessage?.(data);

        // Route to specific handlers based on type
        switch (data.type) {
          case 'position_update':
            onPositionUpdate?.(data.data);
            break;
          case 'pnl_update':
            onPnlUpdate?.(data.data);
            break;
          case 'price_update':
            onPriceUpdate?.(data.data);
            break;
          case 'alert':
            onAlert?.(data.data);
            break;
          case 'metrics_update':
            onMetricsUpdate?.(data.data);
            break;
        }
      });

      // Handle specific event channels
      socketRef.current.on('position_updated', (position: WebSocketPosition) => {
        onPositionUpdate?.(position);
      });

      socketRef.current.on('pnl_updated', (pnl: any) => {
        onPnlUpdate?.(pnl);
      });

      socketRef.current.on('price_updated', (price: any) => {
        onPriceUpdate?.(price);
      });

      socketRef.current.on('alert', (alert: WebSocketAlert) => {
        onAlert?.(alert);
      });

      socketRef.current.on('metrics_updated', (metrics: any) => {
        onMetricsUpdate?.(metrics);
      });

      socketRef.current.on('disconnect', () => {
        console.log('WebSocket disconnected');
        onDisconnect?.();
      });

      socketRef.current.on('error', (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      });
    } catch (err) {
      console.error('Failed to connect WebSocket:', err);
      onError?.(err);
    }
  }, [enabled, onPositionUpdate, onPnlUpdate, onPriceUpdate, onAlert, onMetricsUpdate, onMessage, onConnect, onDisconnect, onError]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
  }, []);

  const send = useCallback((event: string, data?: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data);
    }
  }, []);

  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, connect, disconnect]);

  return {
    socket: socketRef.current,
    isConnected: socketRef.current?.connected || false,
    send,
    disconnect,
  };
}
