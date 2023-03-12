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
    

    
    # Define translation mappings
    Generate = load_model.prediction(input_text, load_model.model)
    
    # Translate the input text if possible
    # output_text = translations.get(input_text, "Translation not found.")
    
    return render_template('test_web.html', output_text=Generate)

if __name__ == '__main__':
    app.run(debug=True)