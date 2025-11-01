import { useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

interface UseWebSocketOptions {
  enabled?: boolean;
  onMessage?: (data: any) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: any) => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    enabled = true,
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

      socketRef.current.on('message', (data) => {
        onMessage?.(data);
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
  }, [enabled, onMessage, onConnect, onDisconnect, onError]);

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
