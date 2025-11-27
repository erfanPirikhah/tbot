"""
Analysis and export module for diagnostic trading system
Extracts data from MongoDB and exports to JSON/CSV formats for detailed analysis
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import logging
from utils.logger import get_mongo_collection

logger = logging.getLogger(__name__)

class DiagnosticDataExporter:
    """Export diagnostic data from MongoDB to JSON/CSV formats for analysis"""
    
    def __init__(self):
        self.collections = {
            'backtest_logs': get_mongo_collection('backtest_logs'),
            'trade_results': get_mongo_collection('trade_results'), 
            'market_snapshots': get_mongo_collection('market_snapshots'),
            'test_metadata': get_mongo_collection('test_metadata'),
            'performance_metrics': get_mongo_collection('performance_metrics')
        }
        
        # Verify all collections are accessible
        for name, collection in self.collections.items():
            if collection is None:
                logger.warning(f"Collection {name} is not accessible")
    
    def get_all_data(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve all diagnostic data, optionally filtered by test_id"""
        data = {}

        for collection_name, collection in self.collections.items():
            if collection is None:
                data[collection_name] = []
                continue

            query = {}
            if test_id:
                query['test_id'] = test_id

            try:
                docs = list(collection.find(query))
                # Convert ObjectId and datetime objects to strings
                docs = self._serialize_documents(docs)
                data[collection_name] = docs
                logger.info(f"Retrieved {len(docs)} records from {collection_name}")
            except Exception as e:
                logger.error(f"Error retrieving data from {collection_name}: {e}")
                data[collection_name] = []

        return data
    
    def _serialize_documents(self, docs: List[Dict]) -> List[Dict]:
        """Convert MongoDB documents to JSON-serializable format"""
        import copy
        from bson import ObjectId
        
        def convert_value(obj):
            if isinstance(obj, ObjectId):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_value(v) for v in obj]
            else:
                return obj
        
        return [convert_value(doc) for doc in docs]
    
    def export_to_json(self, test_id: Optional[str] = None, output_dir: str = "analysis_exports") -> str:
        """Export all data to JSON format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostic_analysis_{timestamp}"
        if test_id:
            filename = f"diagnostic_analysis_{test_id}_{timestamp}"
        
        data = self.get_all_data(test_id)
        
        # Save combined data
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"JSON export completed: {json_path}")
        
        # Also save individual collections
        for collection_name, docs in data.items():
            individual_path = os.path.join(output_dir, f"{filename}_{collection_name}.json")
            with open(individual_path, 'w', encoding='utf-8') as f:
                json.dump(docs, f, ensure_ascii=False, indent=2, default=str)
        
        return json_path
    
    def export_to_csv(self, test_id: Optional[str] = None, output_dir: str = "analysis_exports") -> Dict[str, str]:
        """Export data to CSV format (per collection)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"diagnostic_analysis_{timestamp}"
        if test_id:
            base_filename = f"diagnostic_analysis_{test_id}_{timestamp}"
        
        data = self.get_all_data(test_id)
        csv_paths = {}
        
        for collection_name, docs in data.items():
            if not docs:
                continue
                
            try:
                # Convert to DataFrame
                df = pd.json_normalize(docs)
                
                # Save to CSV
                csv_path = os.path.join(output_dir, f"{base_filename}_{collection_name}.csv")
                df.to_csv(csv_path, index=False)
                csv_paths[collection_name] = csv_path
                logger.info(f"CSV export for {collection_name} completed: {csv_path}")
                
            except Exception as e:
                logger.error(f"Error exporting {collection_name} to CSV: {e}")
        
        return csv_paths
    
    def get_test_ids(self) -> List[str]:
        """Get list of all test IDs available in the database"""
        if self.collections['test_metadata'] is None:
            return []

        try:
            docs = list(self.collections['test_metadata'].find({}, {'test_id': 1}))
            return [doc['test_id'] for doc in docs if 'test_id' in doc]
        except Exception as e:
            logger.error(f"Error getting test IDs: {e}")
            return []
    
    def get_performance_summary(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for analysis"""
        # Get performance metrics
        perf_collection = self.collections['performance_metrics']
        if perf_collection is None:
            return {}

        query = {}
        if test_id:
            query['test_id'] = test_id

        try:
            metrics = list(perf_collection.find(query))
            if not metrics:
                return {}

            # Convert to DataFrame for analysis
            df = pd.json_normalize(metrics)
            summary = {
                'total_tests': len(metrics),
                'avg_win_rate': float(df['win_rate'].mean()) if 'win_rate' in df.columns else 0,
                'avg_sharpe_ratio': float(df['sharpe_ratio'].mean()) if 'sharpe_ratio' in df.columns else 0,
                'avg_max_drawdown': float(df['max_drawdown'].mean()) if 'max_drawdown' in df.columns else 0,
                'total_pnl': float(df['total_pnl'].sum()) if 'total_pnl' in df.columns else 0,
                'avg_profit_factor': float(df['profit_factor'].mean()) if 'profit_factor' in df.columns else 0,
                'avg_expectancy': float(df['expectancy'].mean()) if 'expectancy' in df.columns else 0,
                'best_test': df.loc[df['total_pnl'].idxmax()]['test_id'] if 'test_id' in df.columns and 'total_pnl' in df.columns else None,
                'worst_test': df.loc[df['total_pnl'].idxmin()]['test_id'] if 'test_id' in df.columns and 'total_pnl' in df.columns else None,
            }

            return summary
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def export_detailed_analysis(self, test_id: Optional[str] = None, output_dir: str = "analysis_exports") -> Dict[str, str]:
        """Export comprehensive analysis package"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"full_analysis_{timestamp}"
        if test_id:
            filename = f"full_analysis_{test_id}_{timestamp}"
        
        # Get all data
        data = self.get_all_data(test_id)
        
        # Generate performance summary
        summary = self.get_performance_summary(test_id)
        
        # Combine data and summary
        full_analysis = {
            'analysis_timestamp': datetime.now().isoformat(),
            'requested_test_id': test_id,
            'performance_summary': summary,
            'raw_data': data
        }
        
        # Export JSON
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_analysis, f, ensure_ascii=False, indent=2, default=str)
        
        # Export CSVs
        csv_paths = self.export_to_csv(test_id, output_dir)
        
        # Create summary report
        report_path = os.path.join(output_dir, f"{filename}_summary.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Diagnostic Analysis Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Test ID: {test_id or 'All Tests'}\n")
            f.write(f"\nPerformance Summary:\n")
            for key, value in summary.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nExported Files:\n")
            f.write(f"  JSON: {json_path}\n")
            for collection, csv_path in csv_paths.items():
                f.write(f"  {collection}.csv: {csv_path}\n")
        
        logger.info(f"Detailed analysis export completed")
        logger.info(f"JSON export: {json_path}")
        logger.info(f"CSV exports: {list(csv_paths.values())}")
        logger.info(f"Summary report: {report_path}")
        
        result_paths = {
            'json': json_path,
            'csv': csv_paths,
            'report': report_path,
            'summary': summary
        }
        
        return result_paths

def main():
    """Main function to run the data exporter"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export diagnostic trading data')
    parser.add_argument('--test-id', type=str, help='Specific test ID to export (default: all tests)')
    parser.add_argument('--format', type=str, choices=['json', 'csv', 'both'], default='both', 
                       help='Export format (default: both)')
    parser.add_argument('--output-dir', type=str, default='analysis_exports', 
                       help='Output directory (default: analysis_exports)')
    parser.add_argument('--list-tests', action='store_true', help='List all available test IDs')
    parser.add_argument('--summary', action='store_true', help='Show performance summary only')
    
    args = parser.parse_args()
    
    exporter = DiagnosticDataExporter()
    
    if args.list_tests:
        print("Available test IDs:")
        test_ids = exporter.get_test_ids()
        for test_id in test_ids[:10]:  # Show first 10
            print(f"  - {test_id}")
        if len(test_ids) > 10:
            print(f"  ... and {len(test_ids) - 10} more")
        return
    
    if args.summary:
        summary = exporter.get_performance_summary(args.test_id)
        print("Performance Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return
    
    print(f"Exporting diagnostic data...")
    print(f"  Test ID: {args.test_id or 'All Tests'}")
    print(f"  Format: {args.format}")
    print(f"  Output Directory: {args.output_dir}")
    
    if args.format == 'json':
        json_path = exporter.export_to_json(args.test_id, args.output_dir)
        print(f"✅ JSON export completed: {json_path}")
    elif args.format == 'csv':
        csv_paths = exporter.export_to_csv(args.test_id, args.output_dir)
        print(f"✅ CSV exports completed:")
        for collection, path in csv_paths.items():
            print(f"  {collection}: {path}")
    else:  # both
        paths = exporter.export_detailed_analysis(args.test_id, args.output_dir)
        print(f"✅ Detailed analysis export completed:")
        print(f"  JSON: {paths['json']}")
        print(f"  Report: {paths['report']}")
        print(f"  CSV files: {list(paths['csv'].values())}")
        print(f"\nPerformance Summary:")
        for key, value in paths['summary'].items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()