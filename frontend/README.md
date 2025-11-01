# DeFi Neural Network Trading Dashboard Frontend

Next.js 14 frontend for the DeFi Neural Network trading system.

## Features

- ✅ Real-time trading dashboard
- ✅ WebSocket integration for live updates
- ✅ Performance metrics visualization
- ✅ Position tracking
- ✅ Model comparison
- ✅ Settings management
- ✅ Responsive design

## Technology Stack

- **Framework**: Next.js 14 (App Router)
- **UI**: shadcn/ui + Tailwind CSS
- **Charts**: Recharts
- **State**: Zustand + React Query
- **Real-time**: Socket.io-client
- **Styling**: Tailwind CSS

## Installation

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment

```bash
cp .env.local.example .env.local
```

### 3. Start Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
src/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   ├── providers.tsx      # React Query provider
│   ├── login/             # Login page
│   └── dashboard/         # Dashboard page
├── components/             # React components
│   ├── dashboard-header.tsx
│   ├── engine-controls.tsx
│   ├── positions-table.tsx
│   ├── performance-cards.tsx
│   └── equity-curve-chart.tsx
├── hooks/                  # Custom React hooks
│   ├── useApi.ts          # API query hooks
│   └── useWebSocket.ts    # WebSocket hook
├── lib/                    # Utilities
│   └── api-client.ts      # API client
├── stores/                 # Zustand stores
│   ├── auth-store.ts      # Authentication
│   └── trading-store.ts   # Trading state
└── types/                  # TypeScript types
    └── api.ts             # API types
```

## Available Scripts

```bash
# Development
npm run dev

# Build for production
npm run build

# Start production server
npm run start

# Run linter
npm run lint

# Type check
npm run type-check

# Format code
npm run format
```

## API Integration

The frontend connects to the backend API at:
- **Development**: `http://localhost:8000`
- **Production**: Set via `NEXT_PUBLIC_API_URL`

### Environment Variables

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/updates
```

## Component Hierarchy

```
Layout
├── Header
│   ├── Title
│   ├── WebSocket Status
│   └── Logout Button
├── EngineControls
│   ├── Status Display
│   ├── Start Button
│   └── Stop Button
├── PerformanceCards
│   ├── Sharpe Ratio
│   ├── Total Return
│   ├── Win Rate
│   └── Max Drawdown
├── Charts
│   └── EquityCurveChart
└── PositionsTable
    └── Position Rows
```

## State Management

### Zustand Stores

- **authStore**: Authentication state, login/logout
- **tradingStore**: Engine status, positions, errors

### React Query

- Automatic caching and refetching
- 60s stale time for data freshness
- Background refetch every 2-10 seconds

## Real-time Updates

WebSocket connection provides:
- Position updates
- Price changes
- Performance metrics
- System alerts

Connected via `useWebSocket` hook in dashboard.

## Styling

- **Colors**: CSS variables in `globals.css`
- **Tailwind**: Utility-first CSS framework
- **Components**: Pre-built dashboard card utilities
- **Responsive**: Mobile-first design approach

## Authentication

Simple password-based login:

1. Enter password on login page
2. Token stored in localStorage
3. Automatically added to API requests
4. Verifiable via `/auth/verify` endpoint

## Performance Optimization

- Code splitting via Next.js dynamic imports
- Image optimization
- CSS minification
- JavaScript minification
- Font optimization

## Deployment

### Vercel (Recommended)

1. Connect GitHub repository
2. Set environment variables:
   - `NEXT_PUBLIC_API_URL`
   - `NEXT_PUBLIC_WS_URL`
3. Deploy automatically

```bash
vercel deploy
```

### Self-hosted

```bash
npm run build
npm run start
```

## Development Workflow

1. Start backend: `cd ../backend && ./start.sh`
2. Start frontend: `npm run dev`
3. Open dashboard at `http://localhost:3000`
4. Login with password: `admin`

## Troubleshooting

### API Connection Issues

- Ensure backend is running on `http://localhost:8000`
- Check CORS configuration in backend
- Verify environment variables

### WebSocket Connection Failed

- Check WebSocket URL in `.env.local`
- Ensure backend WebSocket endpoint is active
- Check browser console for errors

### Login Not Working

- Verify backend `/auth/login` endpoint is working
- Check password is correct (default: "admin")
- Review API client configuration

## Next Steps

1. **Phase 3**: Add more dashboard components
2. **Phase 4**: Performance analytics page
3. **Phase 5**: Model comparison UI
4. **Phase 6**: Settings page
5. **Phase 7**: Production deployment

## Support

For issues, check:
- Console errors (F12 in browser)
- Backend logs
- Network tab in DevTools
- API documentation: `../backend/API_DOCUMENTATION.md`
