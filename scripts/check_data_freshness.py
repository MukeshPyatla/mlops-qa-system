#!/usr/bin/env python3
"""
Data freshness check script for the LLM-Powered Q&A System.

This script checks if the knowledge base data is fresh and up-to-date.
It's used as part of the automated MLOps pipeline.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logging import setup_logging, get_logger
from utils.config import get_config


def check_data_freshness(data_dir: str, max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Check if the data in the knowledge base is fresh.
    
    Args:
        data_dir: Directory containing collected data
        max_age_hours: Maximum age in hours before data is considered stale
        
    Returns:
        Dictionary containing freshness check results
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return {
            "fresh": False,
            "reason": "Data directory does not exist",
            "last_update": None,
            "age_hours": None
        }
    
    # Find the most recent data file
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        return {
            "fresh": False,
            "reason": "No data files found",
            "last_update": None,
            "age_hours": None
        }
    
    # Get the most recent file
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    file_mtime = datetime.fromtimestamp(latest_file.stat().st_mtime)
    age_hours = (datetime.now() - file_mtime).total_seconds() / 3600
    
    # Check if data is fresh
    is_fresh = age_hours <= max_age_hours
    
    return {
        "fresh": is_fresh,
        "reason": "Data is fresh" if is_fresh else f"Data is {age_hours:.1f} hours old",
        "last_update": file_mtime.isoformat(),
        "age_hours": age_hours,
        "latest_file": latest_file.name
    }


def check_index_freshness(models_dir: str, max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Check if the vector index is fresh.
    
    Args:
        models_dir: Directory containing model files
        max_age_hours: Maximum age in hours before index is considered stale
        
    Returns:
        Dictionary containing index freshness check results
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return {
            "fresh": False,
            "reason": "Models directory does not exist",
            "last_update": None,
            "age_hours": None
        }
    
    # Look for index files
    index_files = list(models_path.glob("faiss_index*"))
    
    if not index_files:
        return {
            "fresh": False,
            "reason": "No index files found",
            "last_update": None,
            "age_hours": None
        }
    
    # Get the most recent index file
    latest_index = max(index_files, key=lambda f: f.stat().st_mtime)
    file_mtime = datetime.fromtimestamp(latest_index.stat().st_mtime)
    age_hours = (datetime.now() - file_mtime).total_seconds() / 3600
    
    # Check if index is fresh
    is_fresh = age_hours <= max_age_hours
    
    return {
        "fresh": is_fresh,
        "reason": "Index is fresh" if is_fresh else f"Index is {age_hours:.1f} hours old",
        "last_update": file_mtime.isoformat(),
        "age_hours": age_hours,
        "latest_index": latest_index.name
    }


def main():
    """Main data freshness check function."""
    parser = argparse.ArgumentParser(description="Check data freshness")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory to check")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory to check")
    parser.add_argument("--max-age-hours", type=int, default=24, help="Maximum age in hours")
    parser.add_argument("--output-file", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = get_logger("check_data_freshness_script")
    
    try:
        logger.info("Starting data freshness check")
        
        # Check data freshness
        data_freshness = check_data_freshness(args.data_dir, args.max_age_hours)
        logger.info("Data freshness check completed", result=data_freshness)
        
        # Check index freshness
        index_freshness = check_index_freshness(args.models_dir, args.max_age_hours)
        logger.info("Index freshness check completed", result=index_freshness)
        
        # Overall freshness
        overall_fresh = data_freshness["fresh"] and index_freshness["fresh"]
        
        results = {
            "overall_fresh": overall_fresh,
            "data_freshness": data_freshness,
            "index_freshness": index_freshness,
            "check_timestamp": datetime.now().isoformat(),
            "max_age_hours": args.max_age_hours
        }
        
        # Log overall result
        if overall_fresh:
            logger.info("Knowledge base is fresh")
        else:
            logger.warning("Knowledge base is stale", results=results)
        
        # Save results if output file specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
        
        # Exit with error if not fresh (for CI/CD pipeline)
        if not overall_fresh:
            logger.error("Data freshness check failed - knowledge base is stale")
            sys.exit(1)
        
        logger.info("Data freshness check completed successfully")
        
    except Exception as e:
        logger.error("Data freshness check failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 