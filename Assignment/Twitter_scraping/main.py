import twint 
from flask import Flask, render_template, request
from tools import predict_function, twiter_scrab
import spacy
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)

@app.route('/', methods=['GET', 'POST'])
def index():
    bad = []
    good = []
    current_tweets = None
    if request.method == 'POST':
        keyword = request.form['text_data']
        tweets = twiter_scrab.scrp_twitter(word=keyword)
        current_tweets = tweets[0:]

        # bad = []
        # good = []
        positive_data = 0
        negative_data = 0
        keylist = []

        keylist = keyword.lower().split(" ") + [keyword.lower().replace(" ","")]+["@"+keyword.lower().replace(" ","")]+["#"]
        

        for tw in current_tweets:
            
            if predict_function.predict_text(tw.tweet,predict_function.model_load):
                positive_data += 1
                for token in nlp(tw.tweet):
                    if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and token.pos_ != 'SYM' and token.text.lower() not in keylist:
                        good.append(token.lemma_.strip())
            else:
                negative_data += 1
                for token in nlp(tw.tweet):
                    if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and token.pos_ != 'SYM' and token.text.lower() not in keylist :
                        bad.append(token.lemma_.strip())         
    
        
        return render_template('index.html', good=good, bad=bad)
    else :
        return render_template('index.html')
    



if __name__ == '__main__':
    app.run(debug=True)
