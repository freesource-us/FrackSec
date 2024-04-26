# web.py
import traceback
from logger import setup_logger
from flask import Flask, render_template, request
from core import analyze_code

app = Flask(__name__)
logger = setup_logger('fracksec_web.log')


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            code = request.form.get('code')
            config_path = 'config.json'  # Path to the configuration file

            analysis_results = analyze_code(code, config_path)
            # ... (rest of the index function) ...

        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return render_template('error.html'), 500

# ... (rest of the web.py module) ...
# Path: core.py
# core.py
import traceback
from logger import setup_logger

logger = setup_logger('fracksec.log')
