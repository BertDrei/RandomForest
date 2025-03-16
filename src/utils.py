import streamlit as st
import pandas as pd

def display_prediction_results(prediction, probability, features_df, model):
    """Display prediction results and insights"""
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
