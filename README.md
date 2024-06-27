# $\textcolor{red}{For\ educational\ purposes\ only\ !!!!!}$

![fracksec logo](https://github.com/freesource-us/FrackSec/assets/165845111/b2356374-f1bf-4870-a0f1-60ca0e3f100d)


### FrackSec Code Analysis Tool
---------------------------

The Fracksec Analysis Tool is a comprehensive Python-based solution designed to analyze codebases for potential vulnerabilities and coding errors. This tool provides a detailed assessment of code complexity, evolution, interactions, data flows, and Pylint scores to calculate the probability of vulnerabilities and identify potential security risks.

### Features
--------
Clone and Analyze Repositories: Automatically clone a GitHub repository and analyze its Python files.

Cyclomatic Complexity Calculation: Evaluate the complexity of the code to identify high-risk areas.

Code Evolution Analysis: Track the evolution of code by counting the number of commits affecting each file.

Commit Message Analysis: Score commits based on the presence of security-related keywords.

Code Interaction and Data Flow Analysis: Assess the interactions between functions and the flow of data within the code.

Pylint Score Calculation: Evaluate code quality using Pylint and identify potential issues.

Vulnerability Detection: Identify potential vulnerabilities based on known patterns and coding practices.

Coding Error Identification: Detect syntax errors, logical errors, and potential runtime exceptions.

Detailed Reporting: Generate CSV reports and detailed text reports summarizing the findings.

Visualization: Plot the distribution of vulnerability probabilities to visualize the overall risk profile.


### Installation
--------
To install the Fracksec Analysis Tool, clone this repository and install the required dependencies: (using a python virtual environment is recommended):

```bash
git clone https://github.com/freesource-us/FrackSec.git
python3 -m venv FrackSec
cd $_
source bin/activate
pip install -r requirements.txt
```

### Usage
--------
To run the analysis on a specific repository, replace the placeholders in the script with your NVD API key and the repository URL, then execute the script:

python fracksec_analysis_tool.py



### Output
--------


Output
The tool generates several outputs:

CSV File: A summary of the analysis results for each file.
Histogram: A plot of the distribution of vulnerability probabilities.
Detailed Report: A text file containing detailed analysis for each file, including potential vulnerabilities and coding errors.


### Example Output
--------

Vulnerability and Code Quality Analysis Report
============================================

Total files analyzed: 347
Average vulnerability probability: 1.53

Top 10 Most Vulnerable Files:
  cloned_repo/test/functional/p2p_segwit.py - Probability: 3.29
  cloned_repo/test/functional/p2p_opportunistic_1p1c.py - Probability: 2.97
  ...

Detailed Analysis:
File: cloned_repo/build_msvc/msvc-autogen.py
Vulnerability Probability: 1.64
Complexity: 17
Evolution: 0
Interactions: 12
Data Flows: 25
Pylint Score: 8.25
...



### Disclaimer
FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY

This tool is intended solely for educational and research purposes. The authors are not responsible for any misuse or damage caused by the tool. Use it responsibly and only on codebases that you have permission to analyze.

### Contributing
--------
Contributions are welcome! Feel free to open an issue or submit a pull request with your improvements or bug fixes.
