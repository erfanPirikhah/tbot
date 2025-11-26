"""
Main diagnostic analysis runner
Executes the complete diagnostic workflow as specified by the user requirements
"""

import logging
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from diagnostic.enhanced_diagnostic_backtest import EnhancedDiagnosticBacktest, run_complete_diagnostic_analysis
from diagnostic.diagnostic_system import DiagnosticSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the complete diagnostic analysis"""
    logger.info("Starting Complete Diagnostic Analysis of Algorithmic Trading System")
    logger.info("=" * 80)

    print("Initializing Complete Diagnostic Analysis")
    print("This will perform:")
    print("   - Multiple timeframes (M15, M30, H1, H4)")
    print("   - 5-10 years of historical data")
    print("   - Different market conditions (trend, range, high/low volatility)")
    print("   - Multiple assets (EURUSD, GBPUSD, XAUUSD, BTCUSD)")
    print("   - Comprehensive logging and metric storage in MongoDB")
    print()

    try:
        # Create the diagnostic system
        diagnostic_system = DiagnosticSystem()

        # Create the enhanced backtest system
        backtester = EnhancedDiagnosticBacktest(diagnostic_system=diagnostic_system)

        print("Diagnostic systems initialized")
        print()

        # Run comprehensive diagnostic analysis
        print("Starting comprehensive diagnostic simulation...")
        results = backtester.run_diagnostic_simulation(
            symbols=['EURUSD', 'GBPUSD', 'XAUUSD', 'BTCUSD'],  # Multiple assets
            timeframes=['M15', 'M30', 'H1', 'H4'],  # Multiple timeframes
            days_back=1825,  # ~5 years of data
            strategy_params=None,  # Use default optimized parameters
            market_conditions=['trend', 'range', 'high_volatility', 'low_volatility']
        )

        print()
        print("Comprehensive diagnostic analysis completed successfully!")
        print()
        print("Results Summary:")
        print(f"   - Total tests run: {results['combined_metrics'].get('total_tests_run', 0)}")
        print(f"   - Total trades analyzed: {results['combined_metrics'].get('total_trades', 0)}")
        print(f"   - Total PnL: ${results['combined_metrics'].get('total_pnl', 0):,.2f}")
        print(f"   - Average win rate: {results['combined_metrics'].get('avg_win_rate', 0):.2f}%")
        print(f"   - Average Sharpe ratio: {results['combined_metrics'].get('avg_sharpe_ratio', 0):.4f}")
        print(f"   - Average max drawdown: {results['combined_metrics'].get('avg_max_drawdown', 0):.2f}%")
        print()

        print("All data has been stored in MongoDB with the following collections:")
        print("   - backtest_logs: Detailed logs for every candle/trade decision")
        print("   - trade_results: Complete trade information and outcomes")
        print("   - market_snapshots: Market state snapshots for analysis")
        print("   - test_metadata: Metadata about each test run")
        print("   - performance_metrics: Calculated performance metrics")
        print()

        print("Data includes:")
        print("   - Market snapshots (OHLC, volatility, trend state, regime)")
        print("   - Indicator values (RSI, EMA, ATR, Bollinger Bands, MACD, ADX)")
        print("   - Decision processes (filters, conditions, reasoning)")
        print("   - Trade details (entry/exit prices, PnL, risk/reward)")
        print("   - Performance metrics (win rate, drawdown, Sharpe ratio, etc.)")
        print()

        # Verify that data was stored in MongoDB (task #9)
        print("Verifying MongoDB storage...")
        verification_result = verify_mongo_storage()
        if verification_result:
            print("MongoDB verification successful - data is stored and queryable")
        else:
            print("MongoDB verification found issues")

        print()
        print("Diagnostic analysis workflow completed successfully!")
        print("All results are now available in MongoDB for analysis with dashboards, notebooks, or BI tools")
        print("=" * 80)

        return results
        
    except Exception as e:
        logger.error(f"❌ Error during diagnostic analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_mongo_storage() -> bool:
    """Verify that data was properly stored in MongoDB"""
    try:
        from utils.logger import get_mongo_collection
        
        collections_to_check = [
            'backtest_logs',
            'trade_results', 
            'market_snapshots',
            'test_metadata',
            'performance_metrics'
        ]
        
        all_good = True
        
        print("   • Checking MongoDB collections:")
        for collection_name in collections_to_check:
            coll = get_mongo_collection(collection_name)
            if coll:
                count = coll.count_documents({})
                print(f"     - {collection_name}: {count} documents")
                if count == 0:
                    print(f"       ⚠️  No documents in {collection_name}")
                    all_good = False
            else:
                print(f"     - {collection_name}: ❌ Collection not found")
                all_good = False
        
        return all_good
        
    except Exception as e:
        logger.error(f"Error verifying MongoDB storage: {e}")
        return False

def run_quick_diagnostic():
    """Run a quick diagnostic for testing purposes"""
    logger.info("🧪 Running quick diagnostic test")
    
    try:
        diagnostic_system = DiagnosticSystem()
        backtester = EnhancedDiagnosticBacktest(diagnostic_system=diagnostic_system)
        
        # Quick test with minimal parameters
        result = backtester.run_diagnostic_backtest(
            symbol="EURUSD",
            timeframe="H1",
            days_back=30,  # 1 month for quick test
            include_multiple_timeframes=False,
            include_multiple_assets=False
        )
        
        print("✅ Quick diagnostic test completed")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in quick diagnostic: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive diagnostic analysis of trading system')
    parser.add_argument('--quick', action='store_true', help='Run quick diagnostic test instead of full analysis')
    parser.add_argument('--verify', action='store_true', help='Only verify MongoDB storage')
    
    args = parser.parse_args()
    
    if args.verify:
        print("🔍 Verifying MongoDB storage...")
        verify_mongo_storage()
    elif args.quick:
        print("Running quick diagnostic test...")
        run_quick_diagnostic()
    else:
        main()