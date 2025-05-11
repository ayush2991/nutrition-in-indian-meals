import streamlit as st
from datetime import timedelta
import pandas as pd
import altair as alt
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Starting the Nutrition in Indian Meals application")

# Load data
@st.cache_data(ttl=timedelta(days=30))
def load_data():
    try:
        logger.info("Loading nutrition data from CSV")
        nutrition_data = pd.read_csv("./data/nutrition-values.csv")
        logger.info(f"Data loaded successfully with {len(nutrition_data)} dishes")
        return nutrition_data
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        st.error("Data files not found. Please check the file paths.")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"An error occurred while loading data: {str(e)}")
        return None

@st.cache_resource
def load_model():
    try:
        logger.info("Loading sentence transformer model")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Failed to load the text similarity model")
        return None

@st.cache_data(ttl=timedelta(hours=1))
def predict_nutrition_from_text(dish_name_input, data, _nutrient_cols, _model, similarity_threshold=0.7):
    """
    Predicts nutrition for a given dish name by finding the closest match using semantic similarity.
    Uses sentence transformers for better semantic understanding.
    """
    if not dish_name_input.strip():
        logger.warning("Empty dish name provided")
        return None, "Please enter a dish name."

    # Preprocess input: convert to lowercase, remove extra spaces
    clean_input = " ".join(dish_name_input.lower().strip().split())
    logger.info(f"Processing query: '{clean_input}'")
    
    try:
        # Create a list of dish names
        dish_names = data["Dish Name"].unique().tolist()
        logger.debug(f"Comparing against {len(dish_names)} unique dishes")
        
        input_embedding = _model.encode([clean_input], convert_to_tensor=True)
        dish_embeddings = _model.encode(dish_names, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = torch.nn.functional.cosine_similarity(input_embedding, dish_embeddings)
        best_match_idx = torch.argmax(similarities)
        score = similarities[best_match_idx].item()
        best_match = dish_names[best_match_idx]
        
        # Convert similarity score to percentage for display
        score_percentage = int(score * 100)

        if score >= similarity_threshold:
            logger.info(f"Found match '{best_match}' for '{dish_name_input}' with score {score_percentage}%")
            predicted_values = data[data["Dish Name"] == best_match][_nutrient_cols].iloc[0]
            prediction_df = pd.DataFrame([predicted_values])
            prediction_df.insert(0, "Dish Name (Predicted)", f"{dish_name_input} (similar to: {best_match})")
            return prediction_df, f"Prediction based on the closest match: '{best_match}' (Similarity: {score_percentage}%)"
        else:
            logger.warning(f"No confident match found for '{dish_name_input}'. Best attempt: '{best_match}' with score {score_percentage}%")
            return None, f"Could not find a confident match for '{dish_name_input}'. Best guess was '{best_match}' (Similarity: {score_percentage}%), but it's below the threshold of {int(similarity_threshold * 100)}%."
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None, f"An error occurred while processing your request: {str(e)}"

st.set_page_config(
    page_title="Nutrition in Indian Meals",
    page_icon="🍽️",
    layout="wide",
)

# Load model at startup
model = load_model()
if model is None:
    st.error("Failed to load the text similarity model. The application may not work correctly.")
    st.stop()

st.title("Nutrition in Indian Meals")
with st.spinner("Loading data..."):
    logger.info("Initializing application data")
    nutrition_data = load_data()
    if nutrition_data is None:
        logger.error("Failed to load nutrition data, stopping application")
        st.stop()
    logger.info("Application data initialized successfully")
    with st.expander("Browse database", expanded=False):
        st.dataframe(nutrition_data, hide_index=True, height=800)

# Calculate average nutrition values
nutrient_types = nutrition_data.columns[1:]
average_nutrition = pd.DataFrame({
    'Dish Name': ['Average'],
    **{col: [nutrition_data[col].mean()] for col in nutrient_types}
})

# Create two columns for the main layout
col1, col2 = st.columns(2)

# Column 1: Nutrition Prediction
with col1:
    st.header("Predict Nutrition Values")
    user_dish_name = st.text_input("Enter the name of a dish:", key="custom_dish_input")
    
    if st.button("Find Nutrition", key="predict_button") and user_dish_name:
        predicted_df, message = predict_nutrition_from_text(user_dish_name, nutrition_data, nutrient_types, model)
        st.info(message)
        if predicted_df is not None:
            # Original wide display (commented out or removed)
            # display_df = predicted_df.copy()
            # format_dict = {col: '{:.2f}' for col in nutrient_types}
            # st.dataframe(
            #     display_df.style.format(format_dict),
            #     hide_index=True,
            #     use_container_width=True
            # )

            # New vertical display
            # Extract the nutrient values as a Pandas Series
            nutrient_values_series = predicted_df[nutrient_types].iloc[0]
            
            # Convert the Series to a DataFrame for vertical display
            vertical_display_df = nutrient_values_series.reset_index()
            vertical_display_df.columns = ['Nutrient', 'Predicted Value'] # Rename columns
            
            # Define formatting for the 'Predicted Value' column
            format_dict_vertical = {'Predicted Value': '{:.2f}'}
            st.dataframe(
                vertical_display_df.style.format(format_dict_vertical),
                hide_index=True,
                use_container_width=True
            )

# Column 2: Dish Comparison
with col2:
    st.header("Compare Dishes")
    meal_name = st.selectbox("Select first dish:", options=nutrition_data["Dish Name"].unique(), key="meal_1", index=107)
    meal_name_2 = st.selectbox("Select second dish (optional):", 
                            options=["None"] + nutrition_data["Dish Name"].unique().tolist(),
                            key="meal_2", index=109)
    
    # Convert "None" to None for proper handling
    meal_name_2 = None if meal_name_2 == "None" else meal_name_2

    if meal_name:
        # Get selected meal data
        meal_data = nutrition_data[nutrition_data["Dish Name"] == meal_name].copy()

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Nutrient': nutrient_types,
            f'{meal_name}': meal_data[nutrient_types].values[0],
            'Average': [nutrition_data[col].mean() for col in nutrient_types]
        })
        if meal_name_2 and meal_name_2 != meal_name:  # Add second meal if selected and different
            meal_data_2 = nutrition_data[nutrition_data["Dish Name"] == meal_name_2].copy()
            comparison_df[f'{meal_name_2}'] = meal_data_2[nutrient_types].values[0]
        
        # Display the comparison
        st.subheader("Nutrition Values Comparison")
        st.dataframe(
            comparison_df.style.format({
                f'{meal_name}': '{:.2f}',
                f'{meal_name_2}': '{:.2f}' if meal_name_2 else None,
                'Average': '{:.2f}',
            }),
            hide_index=True,
            use_container_width=True
        )
        
        # Calculate normalized values for both meals
        for nutrient in nutrient_types:
            avg_value = nutrition_data[nutrient].mean()
            meal_data[nutrient] = meal_data[nutrient] / avg_value
            if meal_name_2 and meal_name_2 != meal_name and 'meal_data_2' in locals():
                meal_data_2[nutrient] = meal_data_2[nutrient] / avg_value
        
        # Create bar chart data for both meals
        melted_meal1 = meal_data.melt(id_vars=['Dish Name'], var_name='Nutrient', value_name='Value').assign(Meal=meal_name)
        
        if meal_name_2 and meal_name_2 != meal_name and 'meal_data_2' in locals():
            melted_meal2 = meal_data_2.melt(id_vars=['Dish Name'], var_name='Nutrient', value_name='Value').assign(Meal=meal_name_2)
            chart_data = pd.concat([melted_meal1, melted_meal2])
        else:
            chart_data = melted_meal1
        
        # Create bar chart
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Nutrient:N', title='Nutrient'),
            y=alt.Y('Value:Q', 
                    title='Value (Relative to Average)',
                    axis=alt.Axis(format='%')),
            color=alt.Color('Meal:N', title='Meal'),
            tooltip=['Nutrient', 'Meal', alt.Tooltip('Value:Q', format='.2%')]
        )
        if meal_name_2 and meal_name_2 != meal_name:  # Apply xOffset only if comparing two meals
            chart = chart.encode(xOffset='Meal:N')

        chart = chart.properties(
            title='Nutrient Content Relative to Average',
            width=800,
            height=400
        )
        
        # Add a reference line at 1.0 (100%)
        reference_line = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(
            strokeDash=[2, 2],
            color='gray'
        ).encode(y='y:Q')
        
        # Combine chart with reference line
        final_chart = (chart + reference_line)
        
        st.altair_chart(final_chart, use_container_width=True)
        
        # Add explanation
        st.caption("Bars indicate nutrient content relative to average. The dashed line represents the average (100%).")