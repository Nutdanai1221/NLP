from PyPDF2 import PdfReader
from spacy.lang.en.stop_words import STOP_WORDS

def preprocessing(sentence, nlp_n):
    
    stopwords = list(STOP_WORDS)
    doc = nlp_n(sentence)
    cleaned_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
            token.pos_ != 'SYM':
                cleaned_tokens.append(token.lemma_.lower().strip())
                
    return " ".join(cleaned_tokens)

def get_skills(text, nlp_n):
    #pass the text to the nlp
    doc = nlp_n(text)  #note that this nlp already know skills
    education = []
    skills = []
    check_major = False
    count = 0
    old_word = []
    #look at the ents
    for ent in doc.ents:
        #if the ent.label_ is SKILL, then we append to some list
        if ent.label_ == "SKILL":
            skills.append(ent.text)
        elif ent.label_ == "EDUCATION" :
            old_word.append(ent.text)
            check_major = True
        if check_major == True  :
            if ent.label_ == "SKILL" and count <3 :
                old_word.append(ent.text)
                education.append(old_word[0]+ " in " + old_word[1])
                count = count+1
            elif count < 3 :
                count = count + 1
            else :
                count = 0
                check_major = False 
                old_word = []
    
    return skills,education    