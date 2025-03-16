import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# Function to train/load the model
def get_model(force_retrain=False):
    """Load existing model or train a new one if not available or retraining is forced
    
    Args:
        force_retrain: If True, retrain model even if it exists
    """
    model_path = 'employment_model.pkl'
    
    if os.path.exists(model_path) and not force_retrain:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Try to use real data first, fall back to synthetic if not available
        csv_path = 'resume_dataset.csv'  # Update this with your actual CSV file name
        if os.path.exists(csv_path):
            st.info("Training model with real data from CSV...")
            return train_model_from_csv(csv_path, model_path)
        else:
            # Generate synthetic training data since we don't have real data
            st.warning("No CSV data found. Training with synthetic data...")
            return train_synthetic_model(model_path)

def train_model_from_csv(csv_path, model_path):
    """Train model using real data from CSV file"""
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)
        
        # Check if the dataset has the expected columns
        required_columns = [
            'education_high_school', 'education_associate', 'education_bachelor', 
            'education_master', 'education_phd', 'years_experience', 'has_internship', 
            'skills_count', 'prog_languages_count', 'major_cs', 'major_engineering', 
            'major_business', 'major_arts', 'major_science', 'major_other'
        ]
        
        # Verify all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if 'employed' not in df.columns:
            st.error("CSV file missing 'employed' target column")
            return train_synthetic_model(model_path)
            
        if missing_columns:
            st.error(f"CSV file missing required columns: {', '.join(missing_columns)}")
            return train_synthetic_model(model_path)
        
        # Prepare features and target
        X = df[required_columns]
        y = df['employed']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        st.write("Model trained with real data. Classification report:")
        st.text(report)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        save_model_info("real")
        save_model_info(os.path.basename(csv_path))
        
        return model
        
    except Exception as e:
        st.error(f"Error training model with CSV data: {str(e)}")
        st.warning("Falling back to synthetic data...")
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
    
    save_model_info("synthetic")
    save_model_info("synthetic_data")
    
    return model

def generate_suggestions(features, probability):
    """Generate improvement suggestions based on features and prediction probability"""
    suggestions = []
    
    if probability < 0.7:
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
    
    return suggestions

def get_active_dataset_info():
    """Get information about the currently active dataset"""
    info_file = 'model_info.json'
    
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Default info if file doesn't exist or can't be read
    return {
        'dataset': 'synthetic data',
        'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def save_model_info(dataset_name):
    """Save information about model training"""
    info_file = 'model_info.json'
    
    info = {
        'dataset': dataset_name,
        'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(info_file, 'w') as f:
        json.dump(info, f)
