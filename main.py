import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os
import re
import google.generativeai as genai
import json
import io

# Import PyPDF2 for PDF handling
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.error("PyPDF2 is required for PDF parsing. Please install with: pip install PyPDF2")
    st.stop()

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
    Extract the following information from this resume and return it as a JSON object:
    1. Education level (assign one: high_school, associate, bachelor, master, phd)
    2. Years of experience (integer)
    3. Technical skills (list all relevant skills)
    4. Has_internship (true/false)
    5. Programming_languages (list all mentioned)
    6. Major_field (assign one: cs, engineering, business, arts, science, other)
    
    Resume:
    {resume_text}
    
    Return only the JSON object with these fields. Do not include any explanations.
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
            return json.loads(json_str)
        else:
            try:
                return json.loads(response_text)
            except:
                st.error("Couldn't parse JSON from Gemini response")
                return None
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

# Function to train/load the model
def get_model():
    """Load existing model or train a new one if not available"""
    model_path = 'employment_model.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Generate synthetic training data since we don't have real data yet
        st.warning("No trained model found. Training with synthetic data...")
        return train_synthetic_model(model_path)

def train_synthetic_model(model_path):
    """Train model with synthetic data for demo purposes"""
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'education_high_school': np.random.randint(0, 2, n_samples),
        'education_associate': np.random.randint(0, 2, n_samples),
        'education_bachelor': np.random.randint(0, 2, n_samples),
        'education_master': np.random.randint(0, 2, n_samples),
        'education_phd': np.random.randint(0, 2, n_samples),
        'years_experience': np.random.randint(0, 15, n_samples),
        'has_internship': np.random.randint(0, 2, n_samples),
        'skills_count': np.random.randint(0, 20, n_samples),
        'prog_languages_count': np.random.randint(0, 10, n_samples),
        'major_cs': np.random.randint(0, 2, n_samples),
        'major_engineering': np.random.randint(0, 2, n_samples),
        'major_business': np.random.randint(0, 2, n_samples),
        'major_arts': np.random.randint(0, 2, n_samples),
        'major_science': np.random.randint(0, 2, n_samples),
        'major_other': np.random.randint(0, 2, n_samples)
    }
    
    # Create synthetic employment outcome (employed or not)
    # Higher education, experience, internships, and skills increase chances
    X = pd.DataFrame(data)
    
    employment_prob = (
        0.1 +
        0.1 * X['education_high_school'] +
        0.2 * X['education_associate'] +
        0.3 * X['education_bachelor'] +
        0.4 * X['education_master'] +
        0.5 * X['education_phd'] +
        0.05 * X['years_experience'] +
        0.2 * X['has_internship'] +
        0.02 * X['skills_count'] +
        0.03 * X['prog_languages_count'] +
        0.3 * X['major_cs'] +  # CS majors have higher employment chances in this model
        0.2 * X['major_engineering']
    )
    
    # Normalize probabilities to [0,1]
    employment_prob = employment_prob / employment_prob.max()
    y = (np.random.random(n_samples) < employment_prob).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    st.write("Model trained with synthetic data. Classification report:")
    st.text(report)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Streamlit app
def main():
    st.title("Student Employment Opportunity Predictor")
    st.write("Upload your resume or enter your skills to predict employment opportunities")
    
    input_method = st.radio("Input Method", ["Upload PDF Resume", "Enter Skills Manually"])
    
    if input_method == "Upload PDF Resume":
        uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
        if uploaded_file:
            try:
                # Extract text from PDF
                resume_text = extract_text_from_pdf(uploaded_file)
                if resume_text and resume_text.strip():
                    st.success("PDF resume uploaded and parsed successfully!")
                    st.write("Extracted Text Preview (first 500 characters):")
                    st.text(resume_text[:500] + "...")
                else:
                    st.warning("Could not extract text from the PDF. It might be scanned or protected.")
                    # Fallback to manual text entry
                    st.info("Please copy and paste your resume content manually:")
                    resume_text = st.text_area("Resume Content", height=300)
                    if not resume_text.strip():
                        return
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                return
        else:
            st.info("No PDF file uploaded yet.")
            return
    else:
        # Manual entry form
        st.subheader("Enter Your Details")
        education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "PhD"])
        experience = st.number_input("Years of Experience", min_value=0, max_value=50)
        skills = st.text_area("Technical Skills (comma separated)")
        has_internship = st.checkbox("Completed internship")
        programming = st.text_area("Programming Languages (comma separated)")
        major = st.selectbox("Field of Study", ["Computer Science", "Engineering", "Business", "Arts", "Science", "Other"])
        
        # Format manual entry as resume-like text for LLM processing
        resume_text = f"""
        Education: {education}
        Experience: {experience} years
        Technical Skills: {skills}
        Internship: {"Yes" if has_internship else "No"}
        Programming Languages: {programming}
        Field of Study: {major}
        """
    
    if st.button("Predict Employment Opportunities"):
        with st.spinner("Processing information..."):
            # Process resume with LLM
            st.write("Processing resume with AI...")
            llm_output = process_resume_with_llm(resume_text)
            
            if llm_output:
                st.write("AI extracted the following information:")
                st.json(llm_output)
                
                # Preprocess for model
                features = preprocess_llm_output(llm_output)
                features_df = pd.DataFrame([features])
                
                # Get model and predict
                model = get_model()
                prediction = model.predict(features_df)[0]
                probability = model.predict_proba(features_df)[0][1]  # Probability of class 1 (employed)
                
                # Show prediction
                st.subheader("Prediction Results")
                if prediction == 1:
                    st.success(f"Employment Likelihood: High ({probability:.1%})")
                    st.balloons()
                else:
                    st.warning(f"Employment Likelihood: Low ({probability:.1%})")
                
                # Feature importance
                st.subheader("Most Important Factors")
                feature_importance = pd.DataFrame({
                    'Feature': features_df.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("Factors influencing your employment opportunities:")
                for idx, row in feature_importance.head(5).iterrows():
                    feature_name = row['Feature'].replace('_', ' ').title()
                    st.write(f"- {feature_name}: {row['Importance']:.2f}")
                
                # Suggestions
                st.subheader("Suggestions to Improve Opportunities")
                if probability < 0.7:
                    suggestions = []
                    
                    if features.get('has_internship', 0) == 0:
                        suggestions.append("Complete an internship in your field")
                    
                    if features.get('skills_count', 0) < 5:
                        suggestions.append("Learn more technical skills relevant to your field")
                    
                    if features.get('prog_languages_count', 0) < 3 and (features.get('major_cs', 0) == 1 or features.get('major_engineering', 0) == 1):
                        suggestions.append("Learn additional programming languages")
                    
                    if features.get('years_experience', 0) < 2:
                        suggestions.append("Gain more practical experience through projects or part-time work")
                    
                    if not suggestions:
                        suggestions.append("Strengthen your existing skills and network within your industry")
                    
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
                else:
                    st.write("Your profile looks strong! Continue to build on your existing strengths.")
            else:
                st.error("Error processing resume information. Please try again.")

if __name__ == "__main__":
    main()