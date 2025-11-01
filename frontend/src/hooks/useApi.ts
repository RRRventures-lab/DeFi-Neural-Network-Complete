import { useQuery, useMutation, UseQueryResult } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';

// Custom hooks for API calls
export function useEngineStatus() {
  return useQuery({
    queryKey: ['engine', 'status'],
    queryFn: () => apiClient.getEngineStatus(),
    refetchInterval: 2000, // Refetch every 2 seconds
  });
}

export function useMetrics() {
  return useQuery({
    queryKey: ['performance', 'metrics'],
    queryFn: () => apiClient.getMetrics(),
    refetchInterval: 5000,
  });
}

export function usePositions() {
  return useQuery({
    queryKey: ['trading', 'positions'],
    queryFn: async () => {
      const response = await apiClient.getPositions();
      return response.positions || [];
    },
    refetchInterval: 2000,
  });
}

export function useEquityCurve(days: number = 30) {
  return useQuery({
    queryKey: ['performance', 'equity-curve', days],
    queryFn: async () => {
      const response = await apiClient.getEquityCurve(days);
      return response.equity_curve || [];
    },
    refetchInterval: 5000,
  });
}

export function useTrades(limit: number = 100) {
  return useQuery({
    queryKey: ['performance', 'trades', limit],
    queryFn: async () => {
      const response = await apiClient.getTrades(limit);
      return response.trades || [];
    },
    refetchInterval: 10000,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ['models', 'list'],
    queryFn: async () => {
      const response = await apiClient.listModels();
      return response.models || [];
    },
  });
}

export function usePredictions(limit: number = 50) {
  return useQuery({
    queryKey: ['models', 'predictions', limit],
    queryFn: async () => {
      const response = await apiClient.getPredictions(limit);
      return response.predictions || [];
    },
    refetchInterval: 5000,
  });
}

export function useStartEngine() {
  return useMutation({
    mutationFn: (mode: 'paper' | 'live') => apiClient.startEngine(mode),
  });
}

export function useStopEngine() {
  return useMutation({
    mutationFn: () => apiClient.stopEngine(),
  });
}

export function useTradingConfig() {
  return useQuery({
    queryKey: ['config', 'trading'],
    queryFn: async () => {
      const response = await apiClient.getTradingConfig();
      return response.config;
    },
  });
}

export function useRiskLimits() {
  return useQuery({
    queryKey: ['config', 'risk-limits'],
    queryFn: async () => {
      const response = await apiClient.getRiskLimits();
      return response.limits;
    },
  });
}
