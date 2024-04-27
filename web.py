import traceback
from logger import setup_logger
from flask import Flask, render_template, request
from core import analyze_code

app = Flask(__name__)
logger = setup_logger('fracksec_web.log')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        code = request.form.get('code')
        if code:
            try:
                config_path = 'config.json'  # Path to the configuration file
                analysis_results = analyze_code(code, config_path)
                return render_template('index.html', analysis_results=analysis_results)
            except Exception as e:
                logger.error(f"Error: {e}")
                logger.error(traceback.format_exc())
                return render_template('error.html'), 500
        else:
            return render_template('index.html', error="No code provided.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)