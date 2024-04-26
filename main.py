import sys

import networkx as nx
from gudhi import SimplexTree

import numpy as np
import argparse
import ast
from Tools.scripts.dutree import display


class AdvancedCodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_scope = []

    def enter_scope(self, name):
        self.current_scope.append(name)

    def exit_scope(self):
        self.current_scope.pop()

    def current_context(self):
        return '.'.join(self.current_scope) if self.current_scope else 'global'

    def visit_FunctionDef(self, node):
        function_name = f"{self.current_context()}.{node.name}"
        self.enter_scope(node.name)
        self.graph.add_node(function_name, type='function', start_line=node.lineno, end_line=node.end_lineno)
        self.generic_visit(node)
        self.exit_scope()

    def visit_ClassDef(self, node):
        class_name = f"{self.current_context()}.{node.name}"
        self.enter_scope(node.name)
        self.graph.add_node(class_name, type='class', start_line=node.lineno, end_line=node.end_lineno)
        self.generic_visit(node)
        self.exit_scope()

    def visit_Call(self, node):
        called_func = node.func.id if isinstance(node.func, ast.Name) else 'Unknown'
        caller_context = self.current_context()
        self.graph.add_edge(caller_context, called_func, type='calls')
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Handle variable assignments within a scope
        variables = [t.id for t in node.targets if isinstance(t, ast.Name)]
        context = self.current_context()
        for var in variables:
            var_name = f"{context}.{var}"
            self.graph.add_node(var_name, type='variable', scope=context)
            self.graph.add_edge(context, var_name, type='defines')
        self.generic_visit(node)

    def analyze_code(self, code):
        tree = ast.parse(code)
        self.visit(tree)
        return self.graph


# Example usage
if __name__ == "__main__":
    code_text = """
class MyClass:
    def method(self):
        print("Hello")

def foo(x):
    if x > 10:
        print(x)
    y = 20
    bar(y)

def bar(y):
    print(y)
"""
    analyzer = AdvancedCodeAnalyzer()
    results_graph = analyzer.analyze_code(code_text)
    print("Graph Nodes:")
    print(results_graph.nodes(data=True))
    print("Graph Edges:")
    print(results_graph.edges(data=True))

import networkx as nx


def prepare_graph_for_tda(graph):
    """
    Prepare the graph for topological data analysis by converting it to a simplex tree.

    Args:
    graph (networkx.Graph): The graph representation of the source code.

    Returns:
    SimplexTree: A simplex tree ready for TDA computations.
    """
    st = SimplexTree()  # Initialize a SimplexTree

    for node in graph.nodes():
        st.insert([node], filtration=0)  # Insert nodes with a filtration value

    for edge in graph.edges():
        st.insert(list(edge), filtration=1)  # Insert edges with a filtration value

    return st


# Example graph construction
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])

# Convert graph to a simplex tree for TDA
simplex_tree = prepare_graph_for_tda(G)

# Display simplex tree entries for verification
print("Simplex Tree Entries:")
for simplex, filtration in simplex_tree.get_simplices():
    print(simplex, filtration)

import networkx as nx


def calculate_betti_numbers(simplex_tree):
    """
    Calculate the Betti numbers of a given simplex tree.

    Args:
    simplex_tree (SimplexTree): A simplex tree representing the topology of a graph.

    Returns:
    dict: A dictionary where keys are dimensions and values are the corresponding Betti numbers.
    """
    result = simplex_tree.persistence()  # Compute the persistence diagram
    betti_numbers = {}

    # Calculate Betti numbers for dimensions found in the persistence diagram
    for interval in simplex_tree.persistence_intervals_in_dimension(0):
        dim = interval[0]
        if dim in betti_numbers:
            betti_numbers[dim] += 1
        else:
            betti_numbers[dim] = 1

    return betti_numbers


# Example usage in your script:
if __name__ == "__main__":
    # Example graph construction, should be replaced with actual graph from code analysis
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])

    # Convert graph to a simplex tree for TDA
    simplex_tree = prepare_graph_for_tda(G)

    # Calculate Betti numbers
    betti_numbers = calculate_betti_numbers(simplex_tree)

    # Display the calculated Betti numbers
    print("Calculated Betti Numbers:")
    for dimension, number in betti_numbers.items():
        print(f"Betti-{dimension}: {number}")

from gudhi import SimplexTree


def estimate_fractal_dimension(simplex_tree, max_dimension, steps=10):
    """
    Estimates the fractal dimension from Betti numbers calculated at various filtration levels.

    Args:
    simplex_tree (SimplexTree): The simplex tree from the TDA analysis.
    max_dimension (int): The maximum dimension to calculate Betti numbers for.
    steps (int): Number of steps to take in the filtration range.

    Returns:
    float: Estimated fractal dimension.
    """
    filtration_range = np.linspace(0, max_dimension, steps)
    betti_numbers = [simplex_tree.betti_number(i) for i in range(steps)]

    # Perform linear regression on the log-log plot of filtration levels and Betti numbers
    logs = np.log(np.array([filtration_range, betti_numbers]))

    # Handle cases where logs may result in -inf due to log(0)
    logs = np.nan_to_num(logs, nan=0.0, posinf=0.0, neginf=0.0)

    # Perform linear regression
    coefficients = np.polyfit(logs[0], logs[1], 1)
    slope = coefficients[0]

    # Plotting for visual inspection
    plt.figure(figsize=(8, 4))
    plt.scatter(logs[0], logs[1], color='blue', label='Log of Betti Numbers vs. Filtration Level')
    plt.plot(logs[0], np.polyval(coefficients, logs[0]), color='red', label=f'Fit Line: slope={slope:.2f}')
    plt.xlabel('Log of Filtration Level')
    plt.ylabel('Log of Betti Numbers')
    plt.title('Fractal Dimension Estimation')
    plt.legend()
    plt.grid(True)
    plt.show()

    return slope


# Example usage, assuming 'simplex_tree' is already populated from your previous TDA step
fractal_dimension = estimate_fractal_dimension(simplex_tree, max_dimension=2)  # Set the max dimension based on your data
print("Estimated Fractal Dimension:", fractal_dimension)

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetection:
    def __init__(self, baseline_fractal_dimensions):
        """
        Initialize the anomaly detection system with baseline fractal dimensions.
        Preprocess and train an Isolation Forest model for each part of the codebase.

        Args:
        baseline_fractal_dimensions (dict): A dictionary with parts of the codebase as keys and lists of
                                            fractal dimensions as values.
        """
        self.models = {}
        self.scalers = {}
        # Train a model for each part of the codebase
        for part, dimensions in baseline_fractal_dimensions.items():
            # Scaling the fractal dimensions
            scaler = StandardScaler()
            dimensions_scaled = scaler.fit_transform(np.array(dimensions).reshape(-1, 1))
            self.scalers[part] = scaler

            # Isolation Forest with tuned parameters
            model = IsolationForest(n_estimators=150, contamination=0.05)  # Adjust based on expected outliers
            model.fit(dimensions_scaled)
            self.models[part] = model

    def detect_anomalies(self, new_data):
        """
        Detect anomalies in new fractal dimension data after preprocessing.

        Args:
        new_data (dict): A dictionary with parts of the codebase as keys and the new fractal dimension as values.

        Returns:
        dict: A dictionary with the same keys indicating if the data point is an anomaly (True for anomaly).
        """
        anomalies = {}
        for part, dimension in new_data.items():
            model = self.models.get(part)
            scaler = self.scalers.get(part)
            # Handle new or unseen parts of the codebase
            if model and scaler:
                dimension_scaled = scaler.transform(np.array([[dimension]]))
                prediction = model.predict(dimension_scaled)
                anomalies[part] = prediction[0] == -1
            else:
                # Consider new parts as needing review or potential anomalies
                anomalies[part] = True
        return anomalies


# Example usage
baseline_data = {
    'functions': [1.5, 1.6, 1.7, 1.55, 1.65],
    'classes': [2.1, 2.2, 2.15, 2.05, 2.0]
}

detector = AnomalyDetection(baseline_data)
new_fractal_dimensions = {'functions': 1.8, 'classes': 2.3}
anomalies = detector.detect_anomalies(new_fractal_dimensions)
print("Anomalies detected:", anomalies)


class AnomalyMapper:
    def __init__(self, graph, fractal_dimensions, anomalies, config=None):
        """
        Initialize the AnomalyMapper with the necessary data.

        Args:
        graph (networkx.Graph): The graph representation of the source code.
        fractal_dimensions (dict): A dictionary containing fractal dimensions for parts of the code.
        anomalies (dict): A dictionary indicating detected anomalies from the AnomalyDetection module.
        config (dict): Configuration for thresholds and other parameters.
        """
        self.graph = graph
        self.fractal_dimensions = fractal_dimensions
        self.anomalies = anomalies
        self.config = config if config is not None else {'severity_threshold': 1.5, 'critical_keywords': ['core', 'security']}

    def map_anomalies_to_code(self):
        """
        Map anomalies to specific parts of the code and assess their potential impact.
        """
        anomaly_details = {}
        for part, is_anomaly in self.anomalies.items():
            if is_anomaly:
                node_data = self.graph.nodes[part]
                severity = self.calculate_severity(part)
                anomaly_details[part] = {
                    'location': f"Lines {node_data['start_line']} to {node_data['end_line']}",
                    'severity': severity,
                    'description': self.generate_description(part, severity)
                }
        return anomaly_details

    def calculate_criticality(self, part):
        """
        Calculate the criticality of a part based on its role, dependencies, and impact.
        """
        critical_multiplier = 1.0
        for keyword in self.config['critical_keywords']:
            if keyword in part:
                critical_multiplier *= 1.5
        return critical_multiplier

    def calculate_severity(self, part):
        """
        Calculate severity based on deviation from average and adjusted by criticality.
        """
        average_dimension = np.mean(list(self.fractal_dimensions.values()))
        deviation = abs(self.fractal_dimensions[part] - average_dimension)
        return deviation * self.calculate_criticality(part)

    def generate_description(self, part, severity):
        """
        Generate a detailed description of the anomaly with implications and remediation advice.
        """
        threshold = self.config['severity_threshold']
        if severity > threshold:
            return f"High severity anomaly detected in {part}. Review, further analysis, and immediate remediation advised."
        else:
            return f"Moderate severity anomaly detected in {part}. Suggest review and cautious monitoring."

    def rank_anomalies(self):
        """
        Rank anomalies by severity.
        """
        anomaly_details = self.map_anomalies_to_code()
        return sorted(anomaly_details.items(), key=lambda x: x[1]['severity'], reverse=True)

def integrate_and_execute(graph, fractal_dimensions, anomalies, config):
    """
    Execute the anomaly mapping and ranking process.
    """
    mapper = AnomalyMapper(graph, fractal_dimensions, anomalies, config)
    return mapper.rank_anomalies()

# Example usage within the application's main control flow.



def store_fractal_dimensions(graph, node_fractal_dimensions):
    """
    Store fractal dimensions in the graph nodes.

    Args:
    graph (networkx.Graph): The graph where nodes will be updated with fractal dimensions.
    node_fractal_dimensions (dict): A dictionary with node identifiers as keys and fractal dimensions as values.
    """
    # Iterate over the graph nodes and assign fractal dimensions
    for node in graph.nodes():
        if node in node_fractal_dimensions:
            graph.nodes[node]['fractal_dimension'] = node_fractal_dimensions[node]
        else:
            # Optionally handle nodes without a calculated dimension
            graph.nodes[node]['fractal_dimension'] = None

# Example usage of the store_fractal_dimensions function
# Assuming you have a dictionary 'node_fractal_dimensions' with your calculated dimensions
node_fractal_dimensions = {
    'MyClass.method': 1.8,  # Example fractal dimensions for nodes
    'foo': 1.2,
    'bar': 1.5
}

# Assuming 'results_graph' is your graph obtained from the analysis
store_fractal_dimensions(results_graph, node_fractal_dimensions)

# Print updated graph to verify the fractal dimensions are stored
print("Graph Nodes with Fractal Dimensions:")
for node, data in results_graph.nodes(data=True):
    print(f"Node: {node}, Fractal Dimension: {data.get('fractal_dimension')}")




import pygraphviz as pgv
from IPython.display import Image

def visualize_graph(graph):
    """
    Visualize the graph with nodes highlighted based on their fractal dimension scores.

    Args:
    graph (networkx.Graph): The graph to be visualized.
    """
    A = pgv.AGraph(strict=True, directed=True)

    # Add nodes and edges to the Graphviz AGraph object
    for node, data in graph.nodes(data=True):
        fractal_dimension = data.get('fractal_dimension', 0)
        # Use a color scheme to represent the fractal dimension
        color = f"{min(255, max(0, int((fractal_dimension - 1) * 255 / 2)))}"  # Scale fractal dimension to 0-255
        fillcolor = f"#ff{color}{color}"
        A.add_node(node, label=f"{node}\nFD: {fractal_dimension:.2f}", style='filled', fillcolor=fillcolor)

    for u, v, data in graph.edges(data=True):
        A.add_edge(u, v)

    # Render the graph to an image file and display it
    A.layout(prog='dot')  # Use 'dot' layout engine for hierarchical structure
    A.draw('graph.png')
    display(Image(filename='graph.png'))

# Example usage
# Assuming 'results_graph' is your final graph with fractal dimensions stored
visualize_graph(results_graph)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def extract_fractal_dimensions(graph):
    """
    Extract fractal dimensions from the graph nodes.

    Args:
    graph (networkx.Graph): The graph containing nodes with stored fractal dimensions.

    Returns:
    dict: A dictionary of node names to fractal dimensions.
    list: List of node names, preserving the order for heatmap labeling.
    """
    dimensions = {}
    labels = []
    for node, data in graph.nodes(data=True):
        if 'fractal_dimension' in data:
            dimensions[node] = data['fractal_dimension']
            labels.append(node)
    return dimensions, labels


def generate_heatmap(graph):
    """
    Generates a heatmap from the fractal dimensions stored in graph nodes.

    Args:
    graph (networkx.Graph): The graph with nodes containing fractal dimensions.
    """
    dimensions, labels = extract_fractal_dimensions(graph)
    data = [[dimensions.get(row, 0) * dimensions.get(col, 0) for col in labels] for row in labels]

    # Convert data to a numpy array for better handling by seaborn
    data_array = np.array(data)
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_array, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="coolwarm")
    plt.title("Codebase Complexity Heatmap")
    plt.xlabel("Code Components")
    plt.ylabel("Code Components")
    plt.show()


def generate_complexity_plot(graph):
    """
    Generates a bar plot for visualizing the complexity across different parts of the codebase, using stored fractal dimensions.

    Args:
    graph (networkx.Graph): The graph containing nodes with fractal dimensions.
    """
    dimensions, labels = extract_fractal_dimensions(graph)
    values = [dimensions[label] for label in labels]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=labels, y=values, palette="viridis")
    plt.title("Fractal Dimension of Codebase Components")
    plt.xlabel("Code Components")
    plt.ylabel("Fractal Dimension")
    plt.xticks(rotation=45)
    plt.show()


# Example usage of the visualization functions
if __name__ == "__main__":
    # Assuming 'results_graph' is your graph obtained from the analysis
    generate_heatmap(results_graph)
    generate_complexity_plot(results_graph)

import networkx as nx

class ResultsPresenter:
    def __init__(self, graph):
        self.graph = graph

    def format_results(self):
        """Formats the results for display or transmission."""
        formatted_results = []
        for node, attributes in self.graph.nodes(data=True):
            formatted_results.append({
                'node': node,
                'type': attributes.get('type', 'Unknown'),
                'start_line': attributes.get('start_line', 'Not available'),
                'end_line': attributes.get('end_line', 'Not available'),
                'fractal_dimension': attributes.get('fractal_dimension', 'Not calculated')
            })
        return formatted_results

    def present_results(self, mode='standalone'):
        """
        Present the formatted results to the user either in a standalone mode or through a web interface.

        Args:
            mode (str): The mode of presentation, 'standalone' or 'web'.
        """
        results = self.format_results()
        if mode == 'standalone':
            # Print results to the console for standalone use
            print("\nAnalysis Results:")
            for result in results:
                print(f"Node: {result['node']}, Type: {result['type']}, Lines: {result['start_line']}-{result['end_line']}, Fractal Dimension: {result['fractal_dimension']}")
        elif mode == 'web':
            # For web mode, format results as JSON to be sent to the frontend
            results_json = json.dumps({'results': results})
            print("Sending results to the web interface...")
            return results_json

# Example usage of the ResultsPresenter class
if __name__ == "__main__":
    # Assuming 'results_graph' is obtained from the AdvancedCodeAnalyzer
    presenter = ResultsPresenter(results_graph)
    # Call the function with 'standalone' for console output or 'web' for web-based output
    presenter.present_results(mode='standalone')  # Change to 'web' if using in a web environment



class ReportGenerator:
    def __init__(self, analysis_results):
        """
        Initializes the report generator with the results from the analysis.

        Args:
        analysis_results (list): A list of dictionaries containing the analysis results.
        """
        self.analysis_results = analysis_results

    def generate_report(self):
        """
        Generates a structured report summarizing the findings and providing recommendations.

        Returns:
        str: A structured report as a string.
        """
        report_lines = ["Code Analysis Report", "====================", ""]
        for result in self.analysis_results:
            report_lines.append(f"Node: {result['node']}")
            report_lines.append(f"Type: {result['type']}")
            report_lines.append(f"Location: Lines {result['start_line']} to {result['end_line']}")
            report_lines.append(f"Fractal Dimension: {result['fractal_dimension']}")
            recommendation = self.provide_recommendations(result)
            report_lines.append(f"Recommendation: {recommendation}")
            report_lines.append("")  # Add a newline for spacing

        return "\n".join(report_lines)

    def provide_recommendations(self, node_data):
        """
        Provides recommendations based on the node data.

        Args:
        node_data (dict): A dictionary containing data of a node from the graph.

        Returns:
        str: A recommendation or insight based on the analysis.
        """
        if node_data['fractal_dimension'] and float(node_data['fractal_dimension']) > 1.5:
            return "Review for potential refactoring due to high complexity."
        else:
            return "No immediate action required."

# Example usage of the ReportGenerator class
if __name__ == "__main__":
    # Assume that 'results' is a list of dictionaries obtained from ResultsPresenter
    results = [{'node': 'MyClass.method', 'type': 'function', 'start_line': '10', 'end_line': '20', 'fractal_dimension': '1.8'},
               {'node': 'foo', 'type': 'function', 'start_line': '25', 'end_line': '35', 'fractal_dimension': '1.2'}]

    report_generator = ReportGenerator(results)
    report = report_generator.generate_report()
    print(report)






import json

class ResultsPresenter:
    def __init__(self, graph):
        """
        Initializes the presenter with the graph data to prepare the analysis results.

        Args:
        graph (networkx.Graph): The graph containing all nodes and their analysis data.
        """
        self.graph = graph

    def format_results(self):
        """
        Formats the analysis results from the graph to prepare them for reporting.

        Returns:
        list: A list of dictionaries, each containing detailed results for a node.
        """
        results = []
        for node, attrs in self.graph.nodes(data=True):
            result = {
                'Node': node,
                'Type': attrs.get('type', 'Unknown'),
                'Start Line': attrs.get('start_line', 'N/A'),
                'End Line': attrs.get('end_line', 'N/A'),
                'Fractal Dimension': attrs.get('fractal_dimension', 'Not calculated')
            }
            results.append(result)
        return results

    def present_results(self, mode='standalone'):
        """
        Presents the formatted results depending on the operational mode (standalone or web).

        Args:
        mode (str): The presentation mode, either 'standalone' for console outputs or 'web' for web integration.
        """
        results = self.format_results()
        if mode == 'standalone':
            for result in results:
                print(f"Node: {result['Node']}, Type: {result['Type']}, Lines: {result['Start Line']}-{result['End Line']}, Fractal Dimension: {result['Fractal Dimension']}")
        elif mode == 'web':
            # Send results as JSON for web-based applications
            return json.dumps(results)

class ReportGenerator:
    def __init__(self, results):
        """
        Initializes the report generator with formatted results.

        Args:
        results (list): A list of dictionaries with formatted results from the ResultsPresenter.
        """
        self.results = results

    def generate_report(self, output_format='text'):
        """
        Generates a structured report from the analysis results.

        Args:
        output_format (str): The format of the output report, e.g., 'text', 'html', 'pdf'.

        Returns:
        str: The generated report in the specified format.
        """
        report_lines = ["Code Analysis Report", "====================", ""]
        for result in self.results:
            report_lines.append(f"Node: {result['Node']}, Type: {result['Type']}, Lines: {result['Start Line']}-{result['End Line']}, Fractal Dimension: {result['Fractal Dimension']}")
            report_lines.append("Recommendation: Review due to complexity." if float(result.get('Fractal Dimension', 0)) > 1.5 else "Recommendation: No immediate action needed.")
            report_lines.append("")  # Blank line for better readability

        if output_format == 'text':
            return "\n".join(report_lines)
        elif output_format in ['html', 'pdf']:
            # Additional handling for HTML or PDF formats can be implemented here
            pass

# Example usage
if __name__ == "__main__":
    # Assuming 'results_graph' is the graph with all the analysis data included
    presenter = ResultsPresenter(results_graph)
    results = presenter.format_results()
    report_gen = ReportGenerator(results)
    report_text = report_gen.generate_report('text')
    print(report_text)

    # If web-based, use:
    # web_results = presenter.present_results('web')
    # Send `web_results` to the frontend via HTTP response


# main.py
# ... (existing code) ...

from cli import run_cli


def run_web_server():
    pass  # Placeholder for web server functionality


def main():
    parser = argparse.ArgumentParser(description='FrackSec - Code Complexity Analysis Tool')
    parser.add_argument('--cli', action='store_true', help='Run the command-line interface')
    parser.add_argument('--web', action='store_true', help='Run the web interface')
    args = parser.parse_args()

    if args.cli:
        run_cli()
    elif args.web:
        run_web_server()
    else:
        print('Please specify either --cli or --web to run the application.')
        sys.exit(1)

if __name__ == '__main__':
    main()