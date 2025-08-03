#!/usr/bin/env python3
"""
Index rebuilding script for the LLM-Powered Q&A System.

This script rebuilds the vector index from collected data and can be used
both manually and as part of the automated MLOps pipeline.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embedding.embedding_pipeline import EmbeddingPipeline
from utils.logging import setup_logging, get_logger
from utils.config import get_config


def load_collected_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load collected data from the data directory.
    
    Args:
        data_dir: Directory containing collected data files
        
    Returns:
        List of document dictionaries
    """
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return documents
    
    # Find all JSON files in the data directory
    json_files = list(data_path.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract documents from the data
            if "data" in data:
                for item in data["data"]:
                    if "content" in item and "metadata" in item:
                        documents.append(item)
            
            logger.info(f"Loaded {len(data.get('data', []))} documents from {json_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to load {json_file}", error=str(e))
            continue
    
    return documents


def main():
    """Main index rebuilding function."""
    parser = argparse.ArgumentParser(description="Rebuild vector index from collected data")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing collected data")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save the index")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing index before rebuilding")
    parser.add_argument("--save-index", action="store_true", default=True, help="Save the new index")
    parser.add_argument("--index-name", type=str, help="Name for the index file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = get_logger("rebuild_index_script")
    
    try:
        logger.info("Starting index rebuild")
        
        # Initialize embedding pipeline
        embedding_pipeline = EmbeddingPipeline()
        
        # Clear existing index if requested
        if args.clear_existing:
            logger.info("Clearing existing index")
            embedding_pipeline.clear_index()
        
        # Load collected data
        logger.info(f"Loading data from {args.data_dir}")
        documents = load_collected_data(args.data_dir)
        
        if not documents:
            logger.error("No documents found to process")
            sys.exit(1)
        
        logger.info(f"Loaded {len(documents)} documents for processing")
        
        # Process documents through embedding pipeline
        logger.info("Processing documents through embedding pipeline")
        results = embedding_pipeline.process_documents(
            documents=documents,
            save_index=args.save_index,
            index_path=args.index_name
        )
        
        # Log results
        logger.info(
            "Index rebuild completed",
            document_count=results.get("document_count", 0),
            chunk_count=results.get("chunk_count", 0),
            embedding_dimension=results.get("embedding_dimension", 0),
            duration_seconds=results.get("duration_seconds", 0),
            index_path=results.get("index_path", "unknown")
        )
        
        # Save index statistics
        stats = embedding_pipeline.get_pipeline_info()
        stats_file = Path(args.output_dir) / "index_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Index statistics saved to {stats_file}")
        logger.info("Index rebuild script completed successfully")
        
    except Exception as e:
        logger.error("Index rebuild failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 