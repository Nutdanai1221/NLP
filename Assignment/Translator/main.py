from flask import Flask, render_template, request
import utils.model_loads as load_model
import torch
app = Flask(__name__)

device = torch.device('cpu')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.form['input-text']
    translated_text = load_model.predict_word(load_model.model , input_text)
    return render_template('index.html', translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)