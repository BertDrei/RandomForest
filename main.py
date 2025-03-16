import streamlit as st
import pandas as pd
import os
import json
from src.resume_processor import extract_text_from_pdf, process_resume_with_llm, preprocess_llm_output, PDF_SUPPORT
from src.model import get_model, generate_suggestions, get_active_dataset_info
from src.utils import display_prediction_results

def dataset_page():
    st.title("Dataset Management")
    
    # Display currently active dataset
    csv_path = 'resume_dataset.csv'
    if os.path.exists(csv_path):
        st.success(f"Currently active dataset: {csv_path}")
        
        # Display dataset stats
        try:
            df = pd.read_csv(csv_path)
            st.write(f"Dataset size: {len(df)} records")
            st.write(f"Employment rate: {df['employed'].mean():.1%}" if 'employed' in df.columns else "Employment data not found")
            
            # Show preview of current dataset
            st.subheader("Preview of current dataset:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading active dataset: {str(e)}")
    else:
        st.warning("No active dataset found. The model will use synthetic data.")
    
    # Upload new dataset section
    st.subheader("Upload New Dataset")
    uploaded_csv = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.success(f"Dataset '{uploaded_csv.name}' uploaded successfully!")
            
            # Save the uploaded CSV
            df.to_csv('resume_dataset.csv', index=False)
            
            # Show preview
            st.write("Preview of uploaded dataset:")
            st.dataframe(df.head())
            
            # Option to train model with this dataset
            if st.button("Train Model with This Dataset"):
                with st.spinner("Training model..."):
                    model = get_model(force_retrain=True)  # Force retraining with new dataset
                    st.success("Model trained successfully with new dataset!")
                    
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

    st.subheader("Model Training Information")
    model_info = get_active_dataset_info()
    st.write(f"Current model was trained using: {model_info['dataset']}")
    st.write(f"Model last trained on: {model_info['trained_on']}")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Resume Analyzer", "Dataset Management"])
    
    if page == "Resume Analyzer":
        st.title("Student Employment Opportunity Predictor")
        st.write("Upload your resume or enter your skills to predict employment opportunities")

        if not PDF_SUPPORT:
            st.error("PyPDF2 is required for PDF parsing. Please install with: pip install PyPDF2")
            st.stop()

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
                    st.subheader("AI Analysis of Your Resume")

                    # Create columns for better organization
                    col1, col2 = st.columns(2)

                    with col1:
                        # Education
                        st.markdown("### 🎓 Education")
                        education_level = llm_output.get('education_level', 'Not specified')
                        if education_level == 'high_school':
                            education_display = 'High School'
                        elif education_level == 'associate':
                            education_display = 'Associate Degree'
                        elif education_level == 'bachelor':
                            education_display = 'Bachelor\'s Degree'
                        elif education_level == 'master':
                            education_display = 'Master\'s Degree'
                        elif education_level == 'phd':
                            education_display = 'PhD'
                        else:
                            education_display = education_level.title()
                            
                        st.markdown(f"**Level**: {education_display}")
                        
                        # Experience
                        st.markdown("### 💼 Experience")
                        years = llm_output.get('years_of_experience', 0)
                        st.markdown(f"**Years**: {years}")
                        
                        # Internship
                        st.markdown("### 🔍 Internship")
                        has_internship = llm_output.get('has_internship', False)
                        if has_internship:
                            st.markdown("✅ **Completed**")
                        else:
                            st.markdown("❌ **Not completed**")

                    with col2:
                        # Major/Field of Study
                        st.markdown("### 🏫 Field of Study")
                        major_field = llm_output.get('major_field', 'Not specified')
                        if major_field == 'cs':
                            major_display = 'Computer Science'
                        elif major_field == 'engineering':
                            major_display = 'Engineering'
                        elif major_field == 'business':
                            major_display = 'Business'
                        elif major_field == 'arts':
                            major_display = 'Arts'
                        elif major_field == 'science':
                            major_display = 'Science'
                        else:
                            major_display = major_field.title()
                            
                        st.markdown(f"**Major**: {major_display}")
                        
                        # Programming Languages
                        st.markdown("### 💻 Programming Languages")
                        prog_languages = llm_output.get('programming_languages', [])
                        if prog_languages:
                            for lang in prog_languages:
                                st.markdown(f"- {lang}")
                            st.markdown(f"**Total**: {len(prog_languages)} languages")
                        else:
                            st.markdown("None specified")

                    # Technical Skills
                    st.markdown("### 🛠️ Technical Skills")
                    tech_skills = llm_output.get('technical_skills', [])

                    # Check if there are skills to display
                    if tech_skills:
                        # Create a grid layout for skills
                        num_cols = 3
                        skills_rows = [tech_skills[i:i + num_cols] for i in range(0, len(tech_skills), num_cols)]
                        
                        for row in skills_rows:
                            cols = st.columns(num_cols)
                            for i, skill in enumerate(row):
                                cols[i].markdown(f"- {skill}")
                        
                        st.markdown(f"**Total**: {len(tech_skills)} skills")
                    else:
                        st.markdown("None specified")

                    # Add a separator before prediction
                    st.markdown("---")

                    # Preprocess for model
                    features = preprocess_llm_output(llm_output)
                    features_df = pd.DataFrame([features])

                    # Get model and predict
                    model = get_model()
                    prediction = model.predict(features_df)[0]
                    probability = model.predict_proba(features_df)[0][1]  # Probability of class 1 (employed)

                    # Display prediction results
                    display_prediction_results(prediction, probability, features_df, model)

                    # Suggestions
                    st.subheader("Suggestions to Improve Opportunities")
                    suggestions = generate_suggestions(features, probability)

                    if probability < 0.7:
                        for suggestion in suggestions:
                            st.write(f"• {suggestion}")
                    else:
                        st.write("Your profile looks strong! Continue to build on your existing strengths.")
                else:
                    st.error("Error processing resume information. Please try again.")
    elif page == "Dataset Management":
        dataset_page()

if __name__ == "__main__":
    main()
