import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def display_prediction_results(prediction, probability, features_df, model):
    """Display prediction results and insights with visualizations and detailed analysis"""
    # Show prediction
    st.subheader("Prediction Results")
    if prediction == 1:
        st.success(f"Employment Likelihood: High ({probability:.1%})")
        st.balloons()
    else:
        st.warning(f"Employment Likelihood: Low ({probability:.1%})")
    
    # Create visualization for employment probability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gauge chart for probability
    gauge_colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99']
    gauge_threshold = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Determine color based on probability
    color_idx = next((i for i, threshold in enumerate(gauge_threshold) if probability <= threshold), 4)
    gauge_color = gauge_colors[color_idx]
    
    # Create gauge chart
    ax1.pie(
        [probability, 1-probability],
        colors=[gauge_color, '#EEEEEE'],
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.4}
    )
    ax1.add_artist(plt.Circle((0, 0), 0.3, fc='white'))
    ax1.text(0, 0, f"{probability:.1%}", ha='center', va='center', fontsize=20)
    ax1.set_title('Employment Probability', pad=20)
    
    # Feature importance visualization
    feature_importance = pd.DataFrame({
        'Feature': features_df.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Display top 5 features
    top_features = feature_importance.head(5)
    
    # Create human-readable feature names
    readable_names = []
    for feature in top_features['Feature']:
        if feature.startswith('education_'):
            readable_names.append('Education: ' + feature.replace('education_', '').title())
        elif feature.startswith('major_'):
            readable_names.append('Field: ' + feature.replace('major_', '').title())
        elif feature == 'years_experience':
            readable_names.append('Years of Experience')
        elif feature == 'has_internship':
            readable_names.append('Internship Experience')
        elif feature == 'skills_count':
            readable_names.append('Number of Skills')
        elif feature == 'prog_languages_count':
            readable_names.append('Programming Languages')
        else:
            readable_names.append(feature.replace('_', ' ').title())
    
    # Create horizontal bar chart
    bars = ax2.barh(
        readable_names,
        top_features['Importance'],
        color='#4169E1'
    )
    
    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                ha='left', va='center')
    
    ax2.set_title('Top 5 Factors Affecting Employment')
    ax2.set_xlim(0, max(top_features['Importance']) * 1.15)  # Add space for labels
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Comprehensive Analysis Section
    st.subheader("In-Depth Analysis")
    
    # Define market demand and salary potential based on major
    market_insights = {
        "cs": {
            "demand": "Very High", 
            "salary": "$70,000-$120,000",
            "growth": "22% (much faster than average)",
            "hot_skills": ["Machine Learning", "Cloud Computing", "Cybersecurity"]
        },
        "engineering": {
            "demand": "High", 
            "salary": "$65,000-$110,000",
            "growth": "7% (average)",
            "hot_skills": ["Renewable Energy", "Automation", "CAD"]
        },
        "business": {
            "demand": "Moderate", 
            "salary": "$55,000-$90,000",
            "growth": "5% (slower than average)",
            "hot_skills": ["Data Analysis", "Project Management", "Digital Marketing"]
        },
        "arts": {
            "demand": "Low to Moderate", 
            "salary": "$40,000-$75,000",
            "growth": "4% (slower than average)",
            "hot_skills": ["UX/UI Design", "Video Production", "Digital Media"]
        },
        "science": {
            "demand": "Moderate to High", 
            "salary": "$60,000-$100,000",
            "growth": "8% (faster than average)",
            "hot_skills": ["Data Science", "Research Methods", "Lab Techniques"]
        },
        "other": {
            "demand": "Varies", 
            "salary": "$45,000-$85,000",
            "growth": "5% (average)",
            "hot_skills": ["Communication", "Critical Thinking", "Project Management"]
        },
    }
    
    # Get major field
    major_field = None
    for col in features_df.columns:
        if col.startswith('major_') and features_df[col].iloc[0] == 1:
            major_field = col.replace('major_', '')
            break
    
    # Get education level
    education_level = None
    for col in features_df.columns:
        if col.startswith('education_') and features_df[col].iloc[0] == 1:
            education_level = col.replace('education_', '')
            break
    
    # Base analysis text
    experience_years = features_df['years_experience'].iloc[0]
    skills_count = features_df['skills_count'].iloc[0]
    prog_count = features_df['prog_languages_count'].iloc[0]
    has_internship = features_df['has_internship'].iloc[0] == 1
    
    # Generate profile strength score (0-100)
    profile_strength = min(100, int(probability * 100) + 
                     (5 if has_internship else 0) + 
                     min(15, experience_years * 3) +
                     min(10, skills_count * 2) +
                     min(10, prog_count * 2))
                     
    # Create the in-depth analysis
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Profile Overview")
            
            # Profile strength meter
            st.markdown(f"**Profile Strength**: {profile_strength}/100")
            st.progress(profile_strength/100)
            
            if probability >= 0.8:
                st.markdown("#### 🌟 **Excellent Employment Prospects**")
                st.markdown("""Your profile shows strong potential for employment opportunities. 
                With your qualifications, you are likely to be competitive in the job market.""")
            elif probability >= 0.6:
                st.markdown("#### ✅ **Good Employment Prospects**")
                st.markdown("""Your profile indicates favorable employment potential.
                While competitive, you have solid qualifications that employers value.""")
            elif probability >= 0.4:
                st.markdown("#### ⚠️ **Moderate Employment Prospects**")
                st.markdown("""Your profile suggests moderate employment potential.
                There are specific areas where improvements could significantly boost your opportunities.""")
            else:
                st.markdown("#### 🔴 **Challenging Employment Prospects**")
                st.markdown("""Your profile may face challenges in the current job market.
                Focused improvements in key areas could substantially increase your employment potential.""")
            
            # Key strengths and weaknesses
            st.markdown("### Key Profile Insights")
            
            strengths = []
            weaknesses = []
            
            # Analyze education
            if education_level in ['master', 'phd']:
                strengths.append(f"Advanced degree ({education_level.title()}) is highly valued by employers")
            elif education_level == 'bachelor':
                strengths.append("Bachelor's degree meets minimum requirements for most positions")
            else:
                weaknesses.append(f"Limited formal education may restrict some opportunities")
            
            # Analyze experience
            if experience_years > 5:
                strengths.append(f"{experience_years} years of experience is a significant advantage")
            elif experience_years > 2:
                strengths.append(f"{experience_years} years of experience is valuable")
            elif experience_years > 0:
                weaknesses.append(f"Limited work experience ({experience_years} years) may need supplementing")
            else:
                weaknesses.append("No professional experience is a significant limitation")
            
            # Analyze internship
            if has_internship:
                strengths.append("Internship experience demonstrates practical workplace skills")
            else:
                weaknesses.append("No internship experience may limit practical skills validation")
            
            # Analyze skills
            if skills_count > 8:
                strengths.append(f"Diverse skill set ({skills_count} skills) shows versatility")
            elif skills_count > 4:
                strengths.append(f"Solid range of skills ({skills_count})")
            elif skills_count > 0:
                weaknesses.append(f"Limited technical skills ({skills_count}) may restrict opportunities")
            else:
                weaknesses.append("No specified technical skills significantly limits opportunities")
            
            # Analyze programming languages
            if prog_count > 4:
                strengths.append(f"Strong programming background ({prog_count} languages)")
            elif prog_count > 2:
                strengths.append(f"Good programming foundation ({prog_count} languages)")
            elif prog_count > 0 and (major_field == 'cs' or major_field == 'engineering'):
                weaknesses.append(f"Limited programming languages ({prog_count}) for technical field")
            
            # Display strengths and weaknesses
            st.markdown("#### Strengths:")
            for strength in strengths:
                st.markdown(f"✅ {strength}")
                
            st.markdown("#### Areas for Improvement:")
            for weakness in weaknesses:
                st.markdown(f"⚠️ {weakness}")
        
        with col2:
            if major_field and major_field in market_insights:
                insight = market_insights[major_field]
                st.markdown("### Market Insights")
                st.markdown(f"**Field**: {major_field.title()}")
                st.markdown(f"**Market Demand**: {insight['demand']}")
                st.markdown(f"**Projected Growth**: {insight['growth']}")
                st.markdown(f"**Salary Range**: {insight['salary']}")
                
                st.markdown("**In-Demand Skills**:")
                for skill in insight['hot_skills']:
                    st.markdown(f"- {skill}")
    
    # Career timeline simulation
    st.subheader("Career Timeline Projection")
    timeline_fig, ax = plt.subplots(figsize=(10, 5))
    
    # Current year
    current_year = datetime.now().year
    years = list(range(current_year, current_year + 11, 2))
    
    # Base growth on profile strength
    if profile_strength >= 80:
        growth_factor = 1.5
        baseline = 0.7
    elif profile_strength >= 60:
        growth_factor = 1.3
        baseline = 0.5
    elif profile_strength >= 40:
        growth_factor = 1.15
        baseline = 0.3
    else:
        growth_factor = 1.08
        baseline = 0.2
        
    # Calculate career progression
    progression = [probability]
    for i in range(1, len(years)):
        next_val = min(0.95, progression[-1] * growth_factor)
        progression.append(next_val)
    
    # Career salary projection (simplified)
    starting_salary = 40000
    if education_level == 'phd':
        starting_salary = 75000
    elif education_level == 'master':
        starting_salary = 65000
    elif education_level == 'bachelor':
        starting_salary = 55000
        
    if major_field == 'cs' or major_field == 'engineering':
        starting_salary *= 1.2
    elif major_field == 'science':
        starting_salary *= 1.1
    
    salaries = [starting_salary]
    for i in range(1, len(years)):
        growth_rate = 1.05 + (progression[i] * 0.1)  # Higher probability = faster salary growth
        salaries.append(salaries[-1] * growth_rate)
    
    # Plot
    ax.plot(years, progression, marker='o', linestyle='-', color='blue', label='Career Advancement')
    
    # Add labels
    for i, (x, y) in enumerate(zip(years, progression)):
        label = f"{y:.0%}"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Add salary information every other point
        if i % 2 == 0:
            salary_label = f"${salaries[i]:,.0f}"
            ax.annotate(salary_label, (x, y), textcoords="offset points", xytext=(0,-15), ha='center')
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Career Progression')
    ax.set_title('Projected Career Growth Over 10 Years')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    st.pyplot(timeline_fig)
    
    # Add interpretation text
    st.markdown("""
    **Timeline Interpretation**: This projection shows potential career growth based on your current profile. 
    The percentages represent career advancement probability, while dollar figures show potential salary progression.
    These projections assume continued skill development and professional growth.
    """)
