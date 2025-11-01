import { create } from 'zustand';
import type { EngineStatus, Position } from '@/types/api';

interface TradingStore {
  engineStatus: EngineStatus | null;
  positions: Position[];
  isLoading: boolean;
  error: string | null;
  
  setEngineStatus: (status: EngineStatus) => void;
  setPositions: (positions: Position[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  resetState: () => void;
}

export const useTradingStore = create<TradingStore>((set) => ({
  engineStatus: null,
  positions: [],
  isLoading: false,
  error: null,
  
  setEngineStatus: (status) => set({ engineStatus: status }),
  setPositions: (positions) => set({ positions }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  resetState: () => set({
    engineStatus: null,
    positions: [],
    isLoading: false,
    error: null,
  }),
}));
