from flask import Flask, render_template, request, jsonify
from core import analyze_code

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        code = request.form.get('code')
        if code:
            try:
                analysis_results = analyze_code(code)
                return jsonify(analysis_results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'No code provided.'}), 400
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)