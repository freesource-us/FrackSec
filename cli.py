import json
import os
import argparse
import traceback
from core import analyze_code
from logger import setup_logger

logger = setup_logger()

def get_code_from_file(file_path):
    # ... (existing get_code_from_file implementation) ...

def present_results(analysis_results):
    # ... (existing present_results implementation) ...

def run_cli():
    try:
        parser = argparse.ArgumentParser(description='FrackSec - Code Complexity Analysis Tool')
        parser.add_argument('file', nargs='?', help='Path to the source code file')
        parser.add_argument('--config', default=None, help='Path to the configuration file')

        args = parser.parse_args()

        if args.file:
            code = get_code_from_file(args.file)
            if code:
                analysis_results = analyze_code(code, args.config)
                present_results(analysis_results)
            else:
                logger.error(f"Error: Failed to read code from file '{args.file}'")
        else:
            logger.error("Error: No source code file provided.")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    run_cli()