from util.parser import get_skills, preprocessing 
from flask import Flask,render_template, request
from util.tool import read_pdf
import spacy
from PyPDF2 import PdfReader
import os 

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Load the PDF file
        file = request.files["pdf_file"]
        # Load the spaCy model
        nlp = spacy.load("en_core_web_sm")
        # load skill
        skill_path = "Assignment/resume_parser/skills.jsonl"
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(skill_path)
        # Convert the PDF to text
        pdf_text = read_pdf(file) #whole file
        # Process the text with spaCy
        doc = nlp(pdf_text)

        # preprocess
        doc = nlp(preprocessing(doc, nlp))
        # Extract skills and education information
        skill,education = get_skills(doc , nlp)
        

        # Return the extracted information
        skilld = "<body><h3>SKILL</h3><p>"
        for num,skillf in enumerate(list(set(skill))) :
            if num+1 < len(list(set(skill))) :   
                skills = skillf + ", "
                skilld += skills
            else :
                skills = skillf
                skilld += skills   
        skilld += "</p></body>"

        educationd = "<body><h3>Education</h3>"
        for education in education :
            educationd += "<p>" + education + "</p>"
        educationd += "</body>"

        return """
    <html>
    <head>
    <link href="css/styles.css" rel="stylesheet" />
    <style>
    h1 {
       font-size: 50 ;
       font-family: "Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
  font-weight: 250;
    }
    h3 {
       font-size: 35px ;
       font-family: "Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
  font-weight: 250;
    }
    p {
        font-size: 20px ;
        font-family: "Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
  font-weight: 40;
    }
    </style>
    <h1 >Resume Parser</h1>
    </head>
    <body style="background-color: blanchedalmond;">
        """ + skilld + educationd+ """
    </body>
    </html>
"""

    return render_template("home.html")

if __name__ == "__main__":
    # print(skills)
    app.run(debug=True)

