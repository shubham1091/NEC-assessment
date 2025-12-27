#!/usr/bin/env python3


import argparse
import sys
from pathlib import Path


from src.config import load_config
from src.pipeline import setup_logging, NECPipeline


def main():
    """Main entry point for pipeline execution"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='NEC Smart Plant Selection ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                          # Run with default config
  python run_pipeline.py --model lasso            # Override model selection
  python run_pipeline.py --config my_config.yaml  # Use custom config file
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['random_forest', 'gradient_boosting', 'lasso'],
        help='Override model type from config (optional)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure {args.config} exists in the current directory.")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Override model if specified
    if args.model:
        config.model_type = args.model
        print(f"Model type overridden to: {args.model}")
    
    # Setup logging
    logger = setup_logging(config.logs_dir, config.verbose)
    
    # Create and run pipeline
    try:
        pipeline = NECPipeline(config)
        pipeline.run()
        return 0
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        print(f"\nERROR: Pipeline execution failed. See logs for details.")
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
