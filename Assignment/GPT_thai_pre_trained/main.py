from flask import Flask, render_template, request
from utils import load_model


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test_web.html')

@app.route('/Generate', methods=['POST'])
def Generate():
    # Get the input text from the form data
    input_text = request.form['input_text']
    if request.form['submit_button'] == 'Greedy':
        Generate = load_model.prediction_greedy(input_text)

    elif request.form['submit_button'] == 'Beam' :    
        Generate = load_model.prediction_beam(input_text)

    
    return render_template('test_web.html', output_text=Generate)

if __name__ == '__main__':
    app.run(debug=True)