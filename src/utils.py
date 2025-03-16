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
    
    # Sector employment potential visualization
    st.subheader("Employment Potential by Sector")
    st.markdown("Based on your profile, here's your employment potential across different sectors:")
    
    # Get profile features to determine sector fit
    has_programming = features_df['prog_languages_count'].iloc[0] > 0
    education_level_score = 0
    for i, level in enumerate(['high_school', 'associate', 'bachelor', 'master', 'phd']):
        if features_df[f'education_{level}'].iloc[0] == 1:
            education_level_score = i
    
    experience_years = features_df['years_experience'].iloc[0]
    skills_count = features_df['skills_count'].iloc[0]
    has_internship = features_df['has_internship'].iloc[0]
    
    # Determine major field
    major_field = None
    for field in ['cs', 'engineering', 'business', 'arts', 'science', 'other']:
        if features_df[f'major_{field}'].iloc[0] == 1:
            major_field = field
            break
    
    # Calculate sector fitness (base probability adjusted by relevant factors)
    sectors = {
        "Technology": 0.0,
        "Finance": 0.0,
        "Healthcare": 0.0,
        "Education": 0.0,
        "Manufacturing": 0.0,
        "Creative Industries": 0.0,
        "Consulting": 0.0,
        "Government": 0.0
    }
    
    # Base all sectors on overall probability
    base_value = probability * 0.5  # Base 50% of score on overall probability
    
    for sector in sectors:
        sectors[sector] = base_value
    
    # Adjust based on major and other factors
    if major_field == 'cs':
        sectors["Technology"] += 0.4
        sectors["Finance"] += 0.2
        sectors["Consulting"] += 0.15
    elif major_field == 'engineering':
        sectors["Technology"] += 0.25
        sectors["Manufacturing"] += 0.3
        sectors["Healthcare"] += 0.1
    elif major_field == 'business':
        sectors["Finance"] += 0.3
        sectors["Consulting"] += 0.25
        sectors["Government"] += 0.1
    elif major_field == 'arts':
        sectors["Creative Industries"] += 0.35
        sectors["Education"] += 0.15
    elif major_field == 'science':
        sectors["Healthcare"] += 0.25
        sectors["Technology"] += 0.15
        sectors["Education"] += 0.2
    else:  # other
        sectors["Government"] += 0.15
        sectors["Education"] += 0.15
    
    # Adjust based on programming languages
    if has_programming:
        prog_count = features_df['prog_languages_count'].iloc[0]
        tech_boost = min(0.3, prog_count * 0.05)
        sectors["Technology"] += tech_boost
        sectors["Finance"] += tech_boost * 0.5
    
    # Adjust based on education level
    education_boost = education_level_score * 0.05
    sectors["Education"] += education_boost
    sectors["Healthcare"] += education_boost * 0.7
    sectors["Government"] += education_boost * 0.5
    
    # Adjust based on experience
    experience_boost = min(0.2, experience_years * 0.03)
    for sector in sectors:
        sectors[sector] += experience_boost
    
    # Normalize to ensure no value exceeds 1.0
    for sector in sectors:
        sectors[sector] = min(0.95, sectors[sector])
        sectors[sector] = max(0.05, sectors[sector])  # Ensure minimum value
    
    # Sort sectors by probability (highest to lowest)
    sorted_sectors = dict(sorted(sectors.items(), key=lambda item: item[1], reverse=True))
    
    # Create horizontal bar chart for sectors
    fig_sectors, ax_sectors = plt.subplots(figsize=(10, 6))
    
    sector_names = list(sorted_sectors.keys())
    sector_values = list(sorted_sectors.values())
    
    # Select top 6 sectors as requested
    sector_names = sector_names[:6]
    sector_values = sector_values[:6]
    
    # Create bars with colors based on value
    colors = plt.cm.RdYlGn(np.array(sector_values))
    
    bars = ax_sectors.barh(sector_names, [v * 100 for v in sector_values], color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax_sectors.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center')
    
    ax_sectors.set_xlabel('Employment Potential (%)')
    ax_sectors.set_title('Top Employment Sectors for Your Profile')
    ax_sectors.set_xlim(0, 100)
    
    # Add a descriptive note
    plt.tight_layout()
    st.pyplot(fig_sectors)
    
    st.markdown("""
    **Note:** This chart shows your potential fit for different sectors based on your profile. 
    The percentages represent relative employment potential, factoring in your education, 
    experience, skills, and major field of study.
    """)
    
    # Show sector insights for top 3 sectors
    st.subheader("Sector Insights")
    
    top_sector_insights = {
        "Technology": {
            "roles": ["Software Developer", "Data Analyst", "IT Support", "Product Manager"],
            "skills": ["Programming", "Data Analysis", "Cloud Services", "Agile Methodologies"]
        },
        "Finance": {
            "roles": ["Financial Analyst", "Investment Banker", "Accountant", "Risk Analyst"],
            "skills": ["Financial Modeling", "Data Analysis", "Regulatory Knowledge", "Excel"]
        },
        "Healthcare": {
            "roles": ["Clinical Researcher", "Health Informatics", "Healthcare Administrator", "Medical Technologist"],
            "skills": ["Electronic Medical Records", "Healthcare Regulations", "Patient Care", "Medical Terminology"]
        },
        "Education": {
            "roles": ["Teacher", "Curriculum Developer", "Education Technologist", "Academic Advisor"],
            "skills": ["Instructional Design", "Student Assessment", "Educational Technology", "Curriculum Development"]
        },
        "Manufacturing": {
            "roles": ["Process Engineer", "Quality Control", "Production Manager", "Supply Chain Analyst"],
            "skills": ["Lean Manufacturing", "Quality Control", "Supply Chain Management", "CAD/CAM"]
        },
        "Creative Industries": {
            "roles": ["Graphic Designer", "Content Creator", "UX/UI Designer", "Marketing Specialist"],
            "skills": ["Design Software", "Content Creation", "User Research", "Visual Communication"]
        },
        "Consulting": {
            "roles": ["Business Analyst", "Management Consultant", "IT Consultant", "Strategy Consultant"],
            "skills": ["Problem Solving", "Client Management", "Business Analysis", "Project Management"]
        },
        "Government": {
            "roles": ["Policy Analyst", "Program Manager", "Research Associate", "Public Administrator"],
            "skills": ["Policy Analysis", "Public Administration", "Regulatory Knowledge", "Grant Writing"]
        }
    }
    
    col1, col2 = st.columns(2)
    
    # Display insights for top 3 sectors
    for i, sector in enumerate(list(sorted_sectors.keys())[:3]):
        if i % 2 == 0:
            with col1:
                st.markdown(f"#### {sector} ({sorted_sectors[sector]:.1%})")
                st.markdown("**Common Roles:**")
                for role in top_sector_insights[sector]["roles"]:
                    st.markdown(f"- {role}")
                st.markdown("**Key Skills:**")
                for skill in top_sector_insights[sector]["skills"]:
                    st.markdown(f"- {skill}")
        else:
            with col2:
                st.markdown(f"#### {sector} ({sorted_sectors[sector]:.1%})")
                st.markdown("**Common Roles:**")
                for role in top_sector_insights[sector]["roles"]:
                    st.markdown(f"- {role}")
                st.markdown("**Key Skills:**")
                for skill in top_sector_insights[sector]["skills"]:
                    st.markdown(f"- {skill}")
