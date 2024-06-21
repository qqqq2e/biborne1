 
import PyPDF2
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
# typing-extensions>=4.6.1
app = Flask(__name__)

nlp = spacy.load('./output/model-best')

# 
data = pd.read_csv('./call_log_202406211525.csv')

df = pd.DataFrame(data)

# Load TF-IDF vectorizer (replace with your actual preprocessing)
with open('tfidf_vectorizer_model.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

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

@app.route('/a', methods=['GET'])
def get_home2():
    return 'hi said mbisd'
   

@app.route('/get_score', methods=['POST'])
def get_data():
    
    cv = extract_information(pdf_to_text(request.files['cv']))
    job = extract_information(pdf_to_text(request.files['job']))
    
    x0 = calculate_similarity(cv[0],job[0])
    x1 = calculate_similarity(cv[1],job[1])
    x2 = calculate_similarity(cv[2],job[2])
    x3 = calculate_similarity(cv[3],job[3])
    x4 = calculate_similarity(cv[4],job[4])

    score = x0*0.3 + x1*0.2  + x3*0.2 + x4*0.3
    data = jsonify({
        "status":200,
        "score":str(score)
    })
    return data



@app.route('/api/get_similarity', methods=['POST'])

def get_similarity():
    # fun = get_top_k_unique_similar_problems()
    req_data = request.get_json()
    input_text = req_data['input_text']

    # Preprocess input text with the loaded vectorizer
    input_tfidf = vectorizer.transform([input_text])

    # Compute cosine similarities
    cosine_similarities = cosine_similarity(input_tfidf, vectorizer.transform(df['problem'])).flatten()

    # Sort and get top 5 similar problems
    top_5_indices = cosine_similarities.argsort()[-10:][::-1]
    top_5_problems = df.iloc[top_5_indices]['problem'].values
    top_5_solutions = df.iloc[top_5_indices]['solution'].values
    top_5_similarities = cosine_similarities[top_5_indices]
   
    top_k_indices = cosine_similarities.argsort()[::-1]  # sort indices in descending order of similarity
    unique_extensions = []
    unique_similarities = []
    seen_extensions = set()
   
    for index in top_k_indices:
        extension = data.iloc[index]['extension_number']
        if extension not in seen_extensions:
            unique_extensions.append(int(extension))
            unique_similarities.append(cosine_similarities[index])
            seen_extensions.add(extension)
        if len(unique_extensions) == 5:
            break

 
    print(1111)
    # Prepare response
    response = {
        'top_5_problems': top_5_problems.tolist() ,
        'top_5_problems_similarities': top_5_similarities .tolist(),
        'top_5_extension_number': unique_extensions  ,
        'top_5_solutions':top_5_solutions.tolist(),
        'top_5_extension_number_similarities': unique_similarities   
    }

    return jsonify(response)

    
















if __name__ == '__main__':
    app.run(debug=True)
