# cli.py
import json
import os
import argparse
from core import analyze_code


def get_code_from_file(file_path):
    """
    Read the source code from a file.

    Args:
        file_path (str): The path to the file containing the source code.

    Returns:
        str: The source code as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        return code
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")


def run_cli() -> object:
    """
    Run the command-line interface for FrackSec.
    """
    parser = argparse.ArgumentParser(description='FrackSec - Code Complexity Analysis Tool')
    parser.add_argument('file', nargs='?', help='Path to the source code file')
    parser.add_argument('--config', default=None, help='Path to the configuration file')

    args = parser.parse_args()

    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as config_file:
                config = json.load(config_file)
        except FileNotFoundError:
            print(f"Error: Configuration file '{args.config}' not found.")
        except Exception as e:
            print(f"Error: {e}")

    if args.file:
        code = get_code_from_file(args.file)
        if code:
            analysis_results = analyze_code(code, config)
            present_results(analysis_results)
    else:
        print("Error: No source code file provided.")


def present_results(analysis_results):
    """
    Present the analysis results in the command-line interface.

    Args:
        analysis_results (dict): A dictionary containing the analysis results.
    """
    graph = analysis_results['graph']
    fractal_dimensions = analysis_results['fractal_dimensions']
    ranked_anomalies = analysis_results['anomalies']

    print("\nAnalysis Results:")
    print("Graph Nodes:")
    for node, data in graph.nodes(data=True):
        print(
            f"Node: {node}, Type: {data.get('type', 'Unknown')}, Lines:"
            f" {data.get('start_line', 'N/A')}-{data.get('end_line', 'N/A')},"
            f" Fractal Dimension: {fractal_dimensions.get(node, 'Not calculated')}")

    print("\nRanked Anomalies:")
    for anomaly, details in ranked_anomalies:
        print(
            f"Anomaly: {anomaly}, Location: {details['location']}, Severity: {details['severity']}, Description:"
            f" {details['description']}")


if __name__ == '__main__':
    run_cli()

# cli.py
    import traceback
    import argparse
    from core import analyze_code
    from logger import setup_logger

logger = setup_logger()


# ... (rest of the cli.py module) ...

def run_cli(traceback=None):
    """
    Run the command-line interface for FrackSec.
    """
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

# ... (rest of the cli.py module) ...
# Path: core.py
# core.py
    import traceback
    from logger import setup_logger

logger = setup_logger()
