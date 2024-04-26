FrackSec Code Analysis Tool
FrackSec is an advanced code analysis tool designed to parse and analyze source code to identify complexity, potential vulnerabilities, and unusual patterns through topological data analysis (TDA) and machine learning techniques. This tool constructs graphs from code, calculates Betti numbers, estimates fractal dimensions, and performs anomaly detection.

Features
Code Parsing: Parses Python code to construct a directed graph of functions, variables, and their interactions.
Graph Analysis: Utilizes NetworkX to create and analyze directed graphs.
Topological Data Analysis: Integrates with the Gudhi library to perform TDA and calculate Betti numbers.
Fractal Dimension Estimation: Estimates fractal dimensions based on TDA results to assess code complexity.
Anomaly Detection: Employs machine learning algorithms to detect deviations from normal complexity patterns.
Visualization: Offers visualization of graphs and complexity metrics to aid in understanding code structure and anomalies.


Installation
To install FrackSec, you will need Python 3.6 or later. Clone this repository and install the required dependencies:


bash
Copy code
git clone https://github.com/yourusername/fracksec.git
cd fracksec
pip install -r requirements.txt
Usage



FrackSec can be run from the command line or integrated into a web application. Here are the basic steps to use it from the command line:

Basic Command Line Execution:
bash
Copy code
python main.py --file path_to_your_code.py
Web Interface:If you prefer to use a web interface, start the Flask server:
bash
Copy code
python web.py
Then, navigate to http://localhost:5000 in your web browser.



Configuration
Modify config.json to adjust the severity thresholds, critical keywords, and baseline data for anomaly detection. For example:

json
Copy code
{
  "severity_threshold": 1.5,
  "critical_keywords": ["core", "security"],
  "baseline_data": {
    "functions": [1.5, 1.6, 1.7],
    "classes": [2.1, 2.2]
  }
}


Contributing
Contributions to FrackSec are welcome! Please consider the following steps:

Fork the repository.
Create a new branch for each feature or improvement.
Submit a pull request with comprehensive description of changes.
