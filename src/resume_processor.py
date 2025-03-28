import streamlit as st
import re
import json
import io
import google.generativeai as genai

# Import PyPDF2 for PDF handling
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Initialize Gemini client with hardcoded API key
def init_gemini():
    api_key = "AIzaSyB7r8xNPUW0X4vhVKObaUqqBjTt-UrRzo4"
    genai.configure(api_key=api_key)
    return True

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Function to extract structured data from resume using Gemini
def process_resume_with_llm(resume_text):
    """Use Gemini to extract structured features from resume text"""
    
    if not init_gemini():
        return None
    
    prompt = f"""
    Extract the following information from this resume and return it as a JSON object with EXACTLY these keys:
    1. education_level (assign one value: high_school, associate, bachelor, master, phd)
    2. years_of_experience (integer number only)
    3. technical_skills (list all relevant skills as an array)
    4. has_internship (boolean: true or false)
    5. programming_languages (list all mentioned as an array)
    6. major_field (assign one value: cs, engineering, business, arts, science, other)
    
    Resume:
    {resume_text}
    
    Return ONLY the JSON object with these exact keys. Do not include any explanations.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text
        # Use regex to extract JSON part or assume the entire response is JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
        else:
            try:
                result = json.loads(response_text)
            except:
                st.error("Couldn't parse JSON from Gemini response")
                return None
                
        # Ensure consistent key naming and default values
        normalized_data = {
            "education_level": result.get("education_level", "not_specified"),
            "years_of_experience": int(result.get("years_of_experience", 0)),
            "technical_skills": result.get("technical_skills", []),
            "has_internship": result.get("has_internship", False),
            "programming_languages": result.get("programming_languages", []),
            "major_field": result.get("major_field", "other")
        }
        
        return normalized_data
    except Exception as e:
        st.error(f"Error processing resume with Gemini: {str(e)}")
        return None

# Function to preprocess LLM output for model input
def preprocess_llm_output(llm_data):
    """Convert LLM JSON output to feature vector for random forest"""
    features = {}
    
    # Education (one-hot encoding)
    education_levels = ['high_school', 'associate', 'bachelor', 'master', 'phd']
    for level in education_levels:
        features[f'education_{level}'] = 1 if llm_data.get('education_level') == level else 0
    
    # Experience
    features['years_experience'] = llm_data.get('years_of_experience', 0)
    
    # Internship
    features['has_internship'] = 1 if llm_data.get('has_internship', False) else 0
    
    # Technical skills count
    features['skills_count'] = len(llm_data.get('technical_skills', []))
    
    # Programming languages count
    features['prog_languages_count'] = len(llm_data.get('programming_languages', []))
    
    # Major field (one-hot encoding)
    major_fields = ['cs', 'engineering', 'business', 'arts', 'science', 'other']
    for field in major_fields:
        features[f'major_{field}'] = 1 if llm_data.get('major_field') == field else 0
    
    return features
