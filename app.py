from flask import Flask, jsonify,request
import PyPDF2
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
nlp = spacy.load('./output/model-best')




def pdf_to_text(pdf_file):
    try:
        # Get the uploaded PDF file from the request
        # pdf_file = request.files['file']
        

        # Check if a file was uploaded
        if pdf_file is None:
            return jsonify({'error': 'No file provided'})

        # # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Initialize an empty string to store the extracted text
        pdf_text = ""

        # Loop through each page and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()

        # Return the extracted text as a response
        return pdf_text

    except Exception as e:
        return jsonify({'error': str(e)})
def extract_information(text):
    try:

        if text is None:
            return jsonify({'error': 'No text provided'})

        # Initialize lists to store extracted information
        EXPERIENCE = []
        DESIGNATION = []
        CERTIFICATION = []
        DEGREE = []
        SKILLS = []

        # Process the text with spaCy
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ == "YEARS OF EXPERIENCE":
                EXPERIENCE.append(ent.text)
            elif ent.label_ == "DESIGNATION":
                DESIGNATION.append(ent.text)
            elif ent.label_ == "CERTIFICATION":
                CERTIFICATION.append(ent.text)
            elif ent.label_ == "DEGREE":
                DEGREE.append(ent.text)
            elif ent.label_ == "SKILLS":
                SKILLS.append(ent.text)

        # Return the extracted information as a JSON response
        result = [
           EXPERIENCE,
            DESIGNATION,
           CERTIFICATION,
           DEGREE,
            SKILLS
        ]

        return result

    except Exception as e:
        return jsonify({'error': str(e)})
def calculate_similarity(list1,list2):

    # Combine the text from the lists into single strings
    cv_text = ' '.join(list1)
    job_description = ' '.join(list2)

    # Process the text with spaCy
    cv_doc = nlp(cv_text)
    job_doc = nlp(job_description)

    # Compute the similarity score between the CV and job description
    similarity_score = cv_doc.similarity(job_doc)

    # Return the similarity score as a JSON response
    return similarity_score  

@app.route('/', methods=['GET'])
def get_home():
    return 'hi said'

   

@app.route('/api', methods=['POST'])
def get_data():

    cv = extract_information(pdf_to_text(request.files['cv']))
    job = extract_information(pdf_to_text(request.files['job']))
    
    x0 = calculate_similarity(cv[0],job[0])
    x1 = calculate_similarity(cv[1],job[1])
    x2 = calculate_similarity(cv[2],job[2])
    x3 = calculate_similarity(cv[3],job[3])
    x4 = calculate_similarity(cv[4],job[4])

    score = x0*0.3 + x1*0.2  + x3*0.2 + x4*0.3
    return str(score)
    
    
















if __name__ == '__main__':
    app.run(debug=True)
