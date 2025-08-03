#!/usr/bin/env python3
"""
Data collection script for the LLM-Powered Q&A System.

This script collects data from configured sources and can be used
both manually and as part of the automated MLOps pipeline.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_collectors.collector_manager import DataCollectorManager
from utils.logging import setup_logging, get_logger
from utils.config import get_config


async def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description="Collect data from configured sources")
    parser.add_argument("--all-sources", action="store_true", help="Collect from all sources")
    parser.add_argument("--source", type=str, help="Collect from specific source")
    parser.add_argument("--save-results", action="store_true", default=True, help="Save results to disk")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh existing data")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = get_logger("collect_data_script")
    
    try:
        logger.info("Starting data collection")
        
        # Initialize collector manager
        collector_manager = DataCollectorManager()
        
        # Validate configuration
        validation = collector_manager.validate_configuration()
        if not validation["valid"]:
            logger.error("Configuration validation failed", errors=validation["errors"])
            sys.exit(1)
        
        if validation["warnings"]:
            logger.warning("Configuration warnings", warnings=validation["warnings"])
        
        # Collect data
        if args.all_sources:
            logger.info("Collecting from all sources")
            results = await collector_manager.collect_all_data(args.save_results)
        elif args.source:
            logger.info(f"Collecting from source: {args.source}")
            results = await collector_manager.collect_from_source(args.source, args.save_results)
        else:
            logger.info("Collecting from all sources (default)")
            results = await collector_manager.collect_all_data(args.save_results)
        
        # Log results
        if isinstance(results, dict) and "summary" in results:
            summary = results["summary"]
            logger.info(
                "Data collection completed",
                total_items=summary.get("total_items", 0),
                successful_collectors=summary.get("successful_collectors", 0),
                failed_collectors=summary.get("failed_collectors", 0),
                duration_seconds=summary.get("duration_seconds", 0)
            )
        else:
            logger.info("Data collection completed", results=results)
        
        logger.info("Data collection script completed successfully")
        
    except Exception as e:
        logger.error("Data collection failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 