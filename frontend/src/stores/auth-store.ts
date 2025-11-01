import { create } from 'zustand';
import { apiClient } from '@/lib/api-client';

interface AuthStore {
  isAuthenticated: boolean;
  token: string | null;
  user: string | null;
  isLoading: boolean;
  error: string | null;
  
  login: (password: string) => Promise<boolean>;
  logout: () => void;
  setError: (error: string | null) => void;
}

export const useAuthStore = create<AuthStore>((set) => ({
  isAuthenticated: false,
  token: null,
  user: null,
  isLoading: false,
  error: null,
  
  login: async (password: string) => {
    set({ isLoading: true, error: null });
    try {
      const response = await apiClient.login(password);
      set({
        isAuthenticated: true,
        token: response.access_token,
        user: 'admin',
        isLoading: false,
      });
      
      // Store token in localStorage
      if (typeof window !== 'undefined') {
        localStorage.setItem('auth_token', response.access_token);
      }
      return true;
    } catch (err: any) {
      set({
        isAuthenticated: false,
        error: err.response?.data?.detail || 'Login failed',
        isLoading: false,
      });
      return false;
    }
  },
  
  logout: () => {
    apiClient.clearToken();
    set({
      isAuthenticated: false,
      token: null,
      user: null,
      error: null,
    });
    
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
  },
  
  setError: (error) => set({ error }),
}));
