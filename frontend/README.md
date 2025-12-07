# Trading Bot Frontend

Modern Next.js dashboard for the AI-powered trading bot.

## Features

- 📊 **Dashboard**: Real-time system status and performance metrics
- 📈 **Market Analysis**: Live price data with ML regime detection
- 🔄 **Backtesting**: Test strategies on historical data
- 💹 **Live Trading**: Control and monitor live trading operations
- 📑 **Reports**: Detailed performance analytics
- ⚙️ **Configuration**: Risk management and ML model settings

## Tech Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Axios
- Recharts
- Lucide React Icons

## Setup

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Configure Environment**:
   Create a `.env.local` file:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Run Development Server**:
   ```bash
   npm run dev
   ```

4. **Open Browser**:
   Navigate to `http://localhost:3000`

## Project Structure

```
app/
├── page.tsx              # Dashboard
├── market/page.tsx       # Market Analysis
├── backtest/page.tsx     # Backtesting
├── trading/page.tsx      # Live Trading
├── reports/page.tsx      # Reports
└── config/page.tsx       # Configuration

components/
├── layout/Sidebar.tsx    # Navigation
└── ui/                   # Reusable components

lib/
├── api.ts                # API client
└── types.ts              # TypeScript types
```

## API Connection

The frontend connects to the FastAPI backend at `http://localhost:8000`.
Make sure the backend is running before starting the frontend.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint
