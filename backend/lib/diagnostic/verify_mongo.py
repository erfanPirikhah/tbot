"""
Simple MongoDB verification script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_mongo_collection
from datetime import datetime
import uuid

def test_mongo_connection():
    """Test MongoDB connection and collections"""
    print("Starting MongoDB verification...")
    
    collections = ['backtest_logs', 'trade_results', 'market_snapshots', 'test_metadata', 'performance_metrics']
    
    all_tests_passed = True
    test_results = {}
    
    for collection_name in collections:
        try:
            collection = get_mongo_collection(collection_name)
            if collection is not None:
                # Try to insert a test document
                test_doc = {
                    'test_id': str(uuid.uuid4()),
                    'timestamp': datetime.now().isoformat(),
                    'test_type': 'connection_test',
                    'collection_name': collection_name,
                    'status': 'active'
                }
                
                result = collection.insert_one(test_doc)
                test_results[collection_name] = {
                    'status': 'OK',
                    'test_doc_id': str(result.inserted_id),
                    'message': f'Collection {collection_name} is accessible and writable'
                }
                print(f"SUCCESS: {collection_name} - Connected and test document inserted")
            else:
                test_results[collection_name] = {
                    'status': 'FAILED',
                    'message': f'Collection {collection_name} is not accessible'
                }
                print(f"FAILED: {collection_name} - Not accessible")
                all_tests_passed = False
        except Exception as e:
            test_results[collection_name] = {
                'status': 'ERROR',
                'message': f'Error accessing {collection_name}: {str(e)}'
            }
            print(f"ERROR: {collection_name} - {str(e)}")
            all_tests_passed = False
    
    print("\n" + "="*60)
    print("MONGODB VERIFICATION RESULTS:")
    print("="*60)
    
    for collection_name, result in test_results.items():
        status = result['status']
        message = result['message']
        print(f"{collection_name:20} | {status:8} | {message}")
    
    print("="*60)
    if all_tests_passed:
        print("VERIFICATION: ALL COLLECTIONS ARE WORKING PROPERLY")
        print("The diagnostic system can successfully store data in MongoDB as required.")
    else:
        print("VERIFICATION: SOME COLLECTIONS HAVE ISSUES")
        print("Please check MongoDB connection and ensure database is running.")
    
    return all_tests_passed

def verify_data_storage():
    """Verify that all required data types can be stored"""
    print("\nVerifying data storage capabilities...")
    
    # Test different data types that will be stored
    test_records = {
        'backtest_logs': {
            'test_id': 'test_123',
            'timestamp': datetime.now(),
            'market_state': {
                'open': 1.2345,
                'high': 1.2350,
                'low': 1.2340,
                'close': 1.2348,
                'volatility_metrics': {'atr': 0.0012, 'std_dev': 0.0025},
                'trend_state': {'direction': 'BULLISH', 'strength': 0.005}
            },
            'indicators': {
                'rsi': 65.4,
                'ema_fast': 1.2342,
                'ema_slow': 1.2320,
                'atr': 0.0012
            },
            'decision': {
                'decision_type': 'ENTRY',
                'action_taken': 'BUY',
                'decision_reason': 'RSI oversold conditions met',
                'filters_passed': ['RSI', 'Trend'],
                'filters_failed': []
            },
            'result': {'portfolio_value': 10000.0}
        },
        'performance_metrics': {
            'test_id': 'test_123',
            'total_trades': 45,
            'winning_trades': 28,
            'win_rate': 62.2,
            'total_pnl': 1245.67,
            'max_drawdown': -8.5,
            'sharpe_ratio': 1.45,
            'profit_factor': 2.1,
            'expectancy': 15.6,
            'max_win_streak': 5,
            'max_loss_streak': 3
        }
    }
    
    try:
        # Try storing test records
        for collection_name, record in test_records.items():
            collection = get_mongo_collection(collection_name)
            if collection is not None:
                result = collection.insert_one(record)
                print(f"SUCCESS: {collection_name} - Can store complex data structure")
            else:
                print(f"FAILED: {collection_name} - Collection not accessible for data storage")
                return False
        
        print("DATA STORAGE VERIFICATION: PASSED")
        return True
        
    except Exception as e:
        print(f"DATA STORAGE VERIFICATION: FAILED - {str(e)}")
        return False

if __name__ == "__main__":
    print("MongoDB Verification Tool")
    print("This verifies that all required collections exist and are accessible.")
    
    connection_ok = test_mongo_connection()
    storage_ok = verify_data_storage() if connection_ok else False
    
    if connection_ok and storage_ok:
        print("\nOVERALL STATUS: VERIFICATION COMPLETE - ALL SYSTEMS READY")
        print("The diagnostic analysis system is ready to store comprehensive trading data as requested.")
    else:
        print("\nOVERALL STATUS: VERIFICATION FAILED - CHECK CONNECTIONS")