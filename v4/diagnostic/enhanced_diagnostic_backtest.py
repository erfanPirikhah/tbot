"""
Enhanced backtest system with comprehensive diagnostic capabilities
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from backtest.enhanced_rsi_backtest_v4 import EnhancedRSIBacktestV4
from diagnostic.diagnostic_system import DiagnosticSystem
from diagnostic.diagnostic_enhanced_rsi_strategy import DiagnosticEnhancedRsiStrategy
from config.parameters import OPTIMIZED_PARAMS_V4

logger = logging.getLogger(__name__)

class EnhancedDiagnosticBacktest(EnhancedRSIBacktestV4):
    """Enhanced backtest with comprehensive diagnostic logging for every decision"""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 commission: float = 0.0003,
                 slippage: float = 0.0001,
                 enable_plotting: bool = True,
                 detailed_logging: bool = True,
                 save_trade_logs: bool = True,
                 output_dir: str = "logs/diagnostic",
                 diagnostic_system: DiagnosticSystem = None):
        super().__init__(initial_capital, commission, slippage, enable_plotting, 
                        detailed_logging, save_trade_logs, output_dir)
        self.diagnostic_system = diagnostic_system or DiagnosticSystem()
    
    def run_diagnostic_backtest(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        days_back: int = 120,
        strategy_params: Dict[str, Any] = None,
        include_multiple_timeframes: bool = False,
        include_multiple_assets: bool = False
    ) -> Dict[str, Any]:
        """Run comprehensive diagnostic backtest with detailed logging"""
        logger.info(f"🚀 Starting comprehensive diagnostic backtest for {symbol} ({timeframe})")
        
        # Use provided parameters or default
        strategy_params = strategy_params or OPTIMIZED_PARAMS_V4
        
        # Create diagnostic system for this run
        self.diagnostic_system = DiagnosticSystem()
        
        # Run the main diagnostic for the specified symbol/timeframe
        result = self.diagnostic_system.run_comprehensive_diagnostic(
            symbol, timeframe, days_back, strategy_params
        )
        
        results = {
            'primary_result': result,
            'additional_analyses': {}
        }
        
        # Run additional analyses if requested
        if include_multiple_timeframes:
            timeframes = ['M15', 'M30', 'H1', 'H4']
            logger.info(f"🔄 Running multi-timeframe analysis for {symbol}")
            multi_tf_result = self.diagnostic_system.run_multiple_timeframe_diagnostic(
                symbol, timeframes, days_back
            )
            results['additional_analyses']['multi_timeframe'] = multi_tf_result
        
        if include_multiple_assets:
            symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'BTCUSD']
            logger.info(f"🔄 Running multi-asset analysis")
            multi_asset_result = self.diagnostic_system.run_multiple_asset_diagnostic(
                symbols, timeframe, days_back
            )
            results['additional_analyses']['multi_asset'] = multi_asset_result
        
        logger.info(f"✅ Comprehensive diagnostic completed. Test ID: {self.diagnostic_system.test_id}")
        
        return results
    
    def run_diagnostic_simulation(
        self,
        symbols: List[str],
        timeframes: List[str],
        days_back: int = 120,
        strategy_params: Dict[str, Any] = None,
        market_conditions: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive diagnostic across multiple dimensions"""
        logger.info("🚀 Starting comprehensive diagnostic simulation")
        
        strategy_params = strategy_params or OPTIMIZED_PARAMS_V4
        market_conditions = market_conditions or ['trend', 'range', 'high_volatility', 'low_volatility']
        
        all_results = {
            'symbols': symbols,
            'timeframes': timeframes,
            'market_conditions': market_conditions,
            'individual_results': {},
            'combined_metrics': {},
            'run_at': datetime.now()
        }
        
        # Run diagnostics across all combinations
        for symbol in symbols:
            all_results['individual_results'][symbol] = {}
            for timeframe in timeframes:
                logger.info(f"📈 Processing {symbol} - {timeframe}")
                
                # Create a new diagnostic system for each combination
                diag_system = DiagnosticSystem()
                
                # Run comprehensive diagnostic
                result = diag_system.run_comprehensive_diagnostic(
                    symbol, timeframe, days_back, strategy_params
                )
                
                all_results['individual_results'][symbol][timeframe] = result
        
        # Calculate combined metrics
        all_results['combined_metrics'] = self._calculate_combined_metrics(
            all_results['individual_results']
        )
        
        logger.info("✅ Comprehensive diagnostic simulation completed")
        return all_results
    
    def _calculate_combined_metrics(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate combined metrics across all results"""
        metrics = {
            'total_tests_run': 0,
            'total_trades': 0,
            'total_pnl': 0,
            'avg_win_rate': 0,
            'avg_sharpe_ratio': 0,
            'avg_max_drawdown': 0,
            'best_performing': [],
            'worst_performing': []
        }
        
        all_metrics = []
        
        for symbol, timeframes in individual_results.items():
            for timeframe, result in timeframes.items():
                metrics['total_tests_run'] += 1
                
                perf_metrics = result.get('performance_metrics', {})
                if perf_metrics:
                    metrics['total_trades'] += perf_metrics.get('total_trades', 0)
                    metrics['total_pnl'] += perf_metrics.get('total_pnl', 0)
                    
                    # Store for averaging calculations
                    all_metrics.append(perf_metrics)
        
        # Calculate averages
        if all_metrics:
            avg_win_rates = [m.get('win_rate', 0) for m in all_metrics if 'win_rate' in m]
            avg_sharpe_ratios = [m.get('sharpe_ratio', 0) for m in all_metrics if 'sharpe_ratio' in m]
            avg_max_drawdowns = [m.get('max_drawdown', 0) for m in all_metrics if 'max_drawdown' in m]
            
            if avg_win_rates:
                metrics['avg_win_rate'] = round(sum(avg_win_rates) / len(avg_win_rates), 2)
            if avg_sharpe_ratios:
                metrics['avg_sharpe_ratio'] = round(sum(avg_sharpe_ratios) / len(avg_sharpe_ratios), 4)
            if avg_max_drawdowns:
                metrics['avg_max_drawdown'] = round(sum(avg_max_drawdowns) / len(avg_max_drawdowns), 2)
        
        # Sort by performance to identify best/worst
        if all_metrics:
            sorted_by_pnl = sorted(all_metrics, key=lambda x: x.get('total_pnl', 0), reverse=True)
            sorted_by_win_rate = sorted(all_metrics, key=lambda x: x.get('win_rate', 0), reverse=True)
            
            metrics['best_performing'] = sorted_by_pnl[:3]  # Top 3
            metrics['worst_performing'] = sorted_by_pnl[-3:]  # Bottom 3
        
        return metrics

def run_complete_diagnostic_analysis():
    """Function to run the complete diagnostic analysis as requested"""
    logger.info("🚀 Starting complete diagnostic analysis of algorithmic trading system")
    
    # Create diagnostic backtest system
    backtester = EnhancedDiagnosticBacktest()
    
    # Define parameters for comprehensive analysis
    symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'BTCUSD']
    timeframes = ['M15', 'M30', 'H1', 'H4']  # 5-10 years of data will be used as specified
    days_back = 1825  # ~5 years of data
    
    # Run comprehensive diagnostic analysis
    results = backtester.run_diagnostic_simulation(
        symbols=symbols,
        timeframes=timeframes,
        days_back=days_back,
        strategy_params=None,  # Use default optimized parameters
        market_conditions=['trend', 'range', 'high_volatility', 'low_volatility']
    )
    
    logger.info("✅ Complete diagnostic analysis finished!")
    logger.info(f"📊 Results stored in MongoDB collections")
    logger.info(f"📈 Total tests run: {results['combined_metrics'].get('total_tests_run', 0)}")
    
    return results

# Example usage function
def run_sample_diagnostic():
    """Run a sample diagnostic as a test"""
    logger.info("🧪 Running sample diagnostic test")
    
    backtester = EnhancedDiagnosticBacktest()
    
    # Test with single symbol and timeframe first
    result = backtester.run_diagnostic_backtest(
        symbol="EURUSD",
        timeframe="H1", 
        days_back=90,  # 3 months for quick test
        include_multiple_timeframes=False,
        include_multiple_assets=False
    )
    
    logger.info("✅ Sample diagnostic test completed")
    return result