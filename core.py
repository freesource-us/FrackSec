import ast
import networkx as nx
import numpy as np
from gudhi import SimplexTree
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import traceback
from logger import setup_logger
import json

logger = setup_logger('fracksec.log')

class AdvancedCodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_scope = []

    def visit_FunctionDef(self, node):
        function_name = f"{'::'.join(self.current_scope)}::{node.name}"
        self.current_scope.append(node.name)
        self.graph.add_node(function_name, type='function', lineno=node.lineno)
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_ClassDef(self, node):
        class_name = f"{'::'.join(self.current_scope)}::{node.name}"
        self.current_scope.append(node.name)
        self.graph.add_node(class_name, type='class', lineno=node.lineno)
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            caller = f"{'::'.join(self.current_scope)}::{node.func.id}"
            self.graph.add_edge(caller, node.func.id)
        self.generic_visit(node)

    def analyze(self, code):
        tree = ast.parse(code)
        self.visit(tree)
        return self.graph

def prepare_graph_for_tda(graph):
    st = SimplexTree()
    for node in graph.nodes():
        st.insert([node])
    for edge in graph.edges():
        st.insert(edge)
    return st

def calculate_betti_numbers(simplex_tree, max_dimension):
    betti_numbers = {}
    for dim in range(max_dimension + 1):
        betti_numbers[dim] = simplex_tree.betti_numbers()[dim]
    return betti_numbers

def estimate_fractal_dimension(simplex_tree, max_dimension):
    betti_numbers = calculate_betti_numbers(simplex_tree, max_dimension)
    dimensions = list(range(max_dimension + 1))
    betti_values = [betti_numbers[dim] for dim in dimensions]
    coeff = np.polyfit(dimensions, np.log(betti_values), 1)
    return coeff[0]

class AnomalyDetection:
    def __init__(self, contamination=0.1, n_estimators=100):
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators)
        self.scaler = StandardScaler()

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class AnomalyMapper:
    def __init__(self, graph, anomaly_scores, severity_threshold=1.5, critical_keywords=None):
        self.graph = graph
        self.anomaly_scores = anomaly_scores
        self.severity_threshold = severity_threshold
        self.critical_keywords = critical_keywords or []

    def map_anomalies(self):
        anomaly_nodes = [node for node, score in self.anomaly_scores.items() if score > self.severity_threshold]
        anomaly_subgraph = self.graph.subgraph(anomaly_nodes)
        return anomaly_subgraph

    def get_critical_anomalies(self):
        critical_anomalies = []
        for node in self.anomaly_scores:
            if any(keyword in node for keyword in self.critical_keywords):
                critical_anomalies.append(node)
        return critical_anomalies

def store_fractal_dimensions(graph, fractal_dimensions):
    for node, dimension in fractal_dimensions.items():
        graph.nodes[node]['fractal_dimension'] = dimension

def load_config(config_path):
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        return {}

def analyze_code(code, config_path=None):
    try:
        config = load_config(config_path)

        analyzer = AdvancedCodeAnalyzer()
        graph = analyzer.analyze(code)

        simplex_tree = prepare_graph_for_tda(graph)
        max_dimension = config.get('analysis', {}).get('max_dimension', 2)
        betti_numbers = calculate_betti_numbers(simplex_tree, max_dimension)
        fractal_dimensions = {node: estimate_fractal_dimension(simplex_tree, max_dimension) for node in graph.nodes()}

        store_fractal_dimensions(graph, fractal_dimensions)

        contamination = config.get('anomaly_detection', {}).get('contamination', 0.1)
        n_estimators = config.get('anomaly_detection', {}).get('n_estimators', 100)
        anomaly_detection = AnomalyDetection(contamination=contamination, n_estimators=n_estimators)
        anomaly_scores = {node: fractal_dimensions[node] for node in graph.nodes()}
        anomaly_detection.fit(np.array(list(anomaly_scores.values())).reshape(-1, 1))
        anomalies = {node: score for node, score in anomaly_scores.items() if anomaly_detection.predict([[score]]) == -1}

        anomaly_mapper = AnomalyMapper(graph, anomaly_scores)
        anomaly_subgraph = anomaly_mapper.map_anomalies()

        analysis_results = {
            'graph': nx.to_dict_of_dicts(graph),
            'fractal_dimensions': fractal_dimensions,
            'anomalies': list(anomaly_subgraph.nodes())
        }
        return analysis_results
    except Exception as e:
        logger.error(f"Error during code analysis: {e}")
        logger.error(traceback.format_exc())
        return {}