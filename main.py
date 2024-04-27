"""
FrackSec - Code Complexity Analysis Tool
"""

import sys
import argparse
import ast
import json
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pygraphviz as pgv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from gudhi import SimplexTree
from IPython.display import Image
from Tools.scripts.dutree import display

class AdvancedCodeAnalyzer(ast.NodeVisitor):
    """
    Advanced Code Analyzer class to traverse and analyze the AST.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_scope = []

    def enter_scope(self, name):
        self.current_scope.append(name)

    def exit_scope(self):
        self.current_scope.pop()

    def current_context(self):
        return '.'.join(self.current_scope) if self.current_scope else 'global'

    def visit_function_def(self, node):
        function_name = f"{self.current_context()}.{node.name}"
        self.enter_scope(node.name)
        self.graph.add_node(function_name, type='function',
                            start_line=node.lineno, end_line=node.end_lineno)
        self.generic_visit(node)
        self.exit_scope()

    def visit_class_def(self, node):
        class_name = f"{self.current_context()}.{node.name}"
        self.enter_scope(node.name)
        self.graph.add_node(class_name, type='class',
                            start_line=node.lineno, end_line=node.end_lineno)
        self.generic_visit(node)
        self.exit_scope()

    def visit_call(self, node):
        called_func = node.func.id if isinstance(node.func, ast.Name) else 'Unknown'
        caller_context = self.current_context()
        self.graph.add_edge(caller_context, called_func, type='calls')
        self.generic_visit(node)

    def visit_assign(self, node):
        variables = [t.id for t in node.targets if isinstance(t, ast.Name)]
        context = self.current_context()
        for var in variables:
            var_name = f"{context}.{var}"
            self.graph.add_node(var_name, type='variable', scope=context)
            self.graph.add_edge(context, var_name, type='defines')
        self.generic_visit(node)

    def analyze_code(self, code):
        """
        Analyze the given code and build the graph representation.
        """
        tree = ast.parse(code)
        self.visit(tree)
        return self.graph

def prepare_graph_for_tda(graph):
    """
    Prepare the graph for topological data analysis by converting it to a simplex tree.
    """
    st = SimplexTree()

    for node in graph.nodes():
        st.insert([node], filtration=0)

    for edge in graph.edges():
        st.insert(list(edge), filtration=1)

    return st

def calculate_betti_numbers(simplex_tree):
    """
    Calculate the Betti numbers of a given simplex tree.
    """
    betti_numbers = {}

    for dimension, _ in simplex_tree.persistence():
        if dimension in betti_numbers:
            betti_numbers[dimension] += 1
        else:
            betti_numbers[dimension] = 1

    return betti_numbers

def estimate_fractal_dimension(simplex_tree, max_dimension, steps=10):
    """
    Estimate the fractal dimension from Betti numbers calculated at various filtration levels.
    """
    filtration_range = np.linspace(0, max_dimension, steps)
    betti_numbers = [simplex_tree.betti_number(i) for i in range(steps)]

    logs = np.log(np.array([filtration_range, betti_numbers]))
    logs = np.nan_to_num(logs, nan=0.0, posinf=0.0, neginf=0.0)

    coefficients = np.polyfit(logs[0], logs[1], 1)
    slope = coefficients[0]

    plt.figure(figsize=(8, 4))
    plt.scatter(logs[0], logs[1], color='blue', label='Log of Betti Numbers vs. Filtration Level')
    plt.plot(logs[0], np.polyval(coefficients, logs[0]), color='red',
             label=f'Fit Line: slope={slope:.2f}')
    plt.xlabel('Log of Filtration Level')
    plt.ylabel('Log of Betti Numbers')
    plt.title('Fractal Dimension Estimation')
    plt.legend()
    plt.grid(True)
    plt.show()

    return slope

class AnomalyDetection:
    """
    Anomaly Detection class to identify anomalies in fractal dimensions.
    """

    def __init__(self, baseline_fractal_dimensions):
        self.models = {}
        self.scalers = {}

        for part, dimensions in baseline_fractal_dimensions.items():
            scaler = StandardScaler()
            dimensions_scaled = scaler.fit_transform(np.array(dimensions).reshape(-1, 1))
            self.scalers[part] = scaler

            model = IsolationForest(n_estimators=150, contamination=0.05)
            model.fit(dimensions_scaled)
            self.models[part] = model

    def detect_anomalies(self, new_data):
        anomalies = {}
        for part, dimension in new_data.items():
            model = self.models.get(part)
            scaler = self.scalers.get(part)

            if model and scaler:
                dimension_scaled = scaler.transform(np.array([[dimension]]))
                prediction = model.predict(dimension_scaled)
                anomalies[part] = prediction[0] == -1
            else:
                anomalies[part] = True
        return anomalies

class AnomalyMapper:
    """
    Anomaly Mapper class to map anomalies to specific parts of the code.
    """

    def __init__(self, graph, fractal_dimensions, anomalies, config=None):
        self.graph = graph
        self.fractal_dimensions = fractal_dimensions
        self.anomalies = anomalies
        self.config = config if config is not None else {
            'severity_threshold': 1.5,
            'critical_keywords': ['core', 'security']
        }

    def map_anomalies_to_code(self):
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
        critical_multiplier = 1.0
        for keyword in self.config['critical_keywords']:
            if keyword in part:
                critical_multiplier *= 1.5
        return critical_multiplier

    def calculate_severity(self, part):
        average_dimension = np.mean(list(self.fractal_dimensions.values()))
        deviation = abs(self.fractal_dimensions[part] - average_dimension)
        return deviation * self.calculate_criticality(part)

    def generate_description(self, part, severity):
        threshold = self.config['severity_threshold']
        if severity > threshold:
            return (f"High severity anomaly detected in {part}. "
                    "Review, further analysis, and immediate remediation advised.")
        else:
            return (f"Moderate severity anomaly detected in {part}. "
                    "Suggest review and cautious monitoring.")

    def rank_anomalies(self):
        anomaly_details = self.map_anomalies_to_code()
        return sorted(anomaly_details.items(), key=lambda x: x[1]['severity'], reverse=True)

def integrate_and_execute(graph, fractal_dimensions, anomalies, config):
    """
    Execute the anomaly mapping and ranking process.
    """
    mapper = AnomalyMapper(graph, fractal_dimensions, anomalies, config)
    return mapper.rank_anomalies()

def store_fractal_dimensions(graph, node_fractal_dimensions):
    """
    Store fractal dimensions in the graph nodes.
    """
    for node in graph.nodes():
        if node in node_fractal_dimensions:
            graph.nodes[node]['fractal_dimension'] = node_fractal_dimensions[node]
        else:
            graph.nodes[node]['fractal_dimension'] = None

def visualize_graph(graph):
    """
    Visualize the graph with nodes highlighted based on their fractal dimension scores.
    """
    a_graph = pgv.AGraph(strict=True, directed=True)

    for node, data in graph.nodes(data=True):
        fractal_dimension = data.get('fractal_dimension', 0)
        color = f"{min(255, max(0, int((fractal_dimension - 1) * 255 / 2)))}"
        fillcolor = f"#ff{color}{color}"
        a_graph.add_node(node, label=f"{node}\nFD: {fractal_dimension:.2f}",
                         style='filled', fillcolor=fillcolor)

    for u, v, data in graph.edges(data=True):
        a_graph.add_edge(u, v)

    a_graph.layout(prog='dot')
    a_graph.draw('graph.png')
    display(Image(filename='graph.png'))

def extract_fractal_dimensions(graph):
    """
    Extract fractal dimensions from the graph nodes.
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
    Generate a heatmap from the fractal dimensions stored in graph nodes.
    """
    dimensions, labels = extract_fractal_dimensions(graph)
    data = [[dimensions.get(row, 0) * dimensions.get(col, 0) for col in labels] for row in labels]

    data_array = np.array(data)
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_array, annot=True, fmt=".2f", xticklabels=labels,
                yticklabels=labels, cmap="coolwarm")
    plt.title("Codebase Complexity Heatmap")
    plt.xlabel("Code Components")
    plt.ylabel("Code Components")
    plt.show()

def generate_complexity_plot(graph):
    """
    Generate a bar plot for visualizing the complexity across different parts of the codebase.
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

class ResultsPresenter:
    """
    Results Presenter class to format and present the analysis results.
    """

    def __init__(self, graph):
        self.graph = graph

    def format_results(self):
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
        results = self.format_results()
        if mode == 'standalone':
            print("\nAnalysis Results:")
            for result in results:
                print(f"Node: {result['node']}, Type: {result['type']}, "
                      f"Lines: {result['start_line']}-{result['end_line']}, "
                      f"Fractal Dimension: {result['fractal_dimension']}")
        elif mode == 'web':
            results_json = json.dumps({'results': results})
            print("Sending results to the web interface...")
            return results_json

class ReportGenerator:
    """
    Report Generator class to generate structured reports from the analysis results.
    """

    def __init__(self, analysis_results):
        self.analysis_results = analysis_results

    def generate_report(self):
        report_lines = ["Code Analysis Report", "====================", ""]
        for result in self.analysis_results:
            report_lines.append(f"Node: {result['node']}")
            report_lines.append(f"Type: {result['type']}")
            report_lines.append(f"Location: Lines {result['start_line']} to {result['end_line']}")
            report_lines.append(f"Fractal Dimension: {result['fractal_dimension']}")
            recommendation = self.provide_recommendations(result)
            report_lines.append(f"Recommendation: {recommendation}")
            report_lines.append("")

        return "\n".join(report_lines)

    def provide_recommendations(self, node_data):
        if node_data['fractal_dimension'] and float(node_data['fractal_dimension']) > 1.5:
            return "Review for potential refactoring due to high complexity."
        else:
            return "No immediate action required."

def run_web_server():
    pass

def main():
    """
    Main function to run the FrackSec code complexity analysis tool.
    """
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