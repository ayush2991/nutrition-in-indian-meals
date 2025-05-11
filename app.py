import streamlit as st
from datetime import timedelta
import pandas as pd
import altair as alt
import logging
from thefuzz import process, fuzz # For string matching

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Starting the application")

# Load data
@st.cache_data(ttl=timedelta(days=30))
def load_data():
    try:
        nutrition_data = pd.read_csv("./data/nutrition-values.csv")
        logger.info("Data loaded successfully")
        return nutrition_data
    except FileNotFoundError:
        st.error("Data files not found. Please check the file paths.")
        return None

st.set_page_config(
    page_title="Nutrition in Indian Meals",
    page_icon="🍽️",
    layout="wide",
)

st.title("Nutrition in Indian Meals")
with st.spinner("Loading data..."):
    nutrition_data = load_data()
    if nutrition_data is None:
        st.stop()
    with st.expander("Browse database", expanded=False):
        st.dataframe(nutrition_data, hide_index=True, height=800)

# Calculate average nutrition values
nutrient_types = nutrition_data.columns[1:]
average_nutrition = pd.DataFrame({
    'Dish Name': ['Average'],
    **{col: [nutrition_data[col].mean()] for col in nutrient_types}
})

# Text input for dish name prediction
user_dish_name = st.text_input("Enter the name of a dish to predict its nutrition:", key="custom_dish_input")

# --- Section for Free Text Dish Name Prediction ---
@st.cache_data(ttl=timedelta(hours=1)) # Cache predictions for a while
def predict_nutrition_from_text(dish_name_input, data, _nutrient_cols, similarity_threshold=70):
    """
    Predicts nutrition for a given dish name by finding the closest match in the dataset.
    Uses fuzzy string matching with preprocessing for better accuracy.
    """
    if not dish_name_input.strip():
        return None, "Please enter a dish name."

    # Preprocess input: convert to lowercase, remove extra spaces
    clean_input = " ".join(dish_name_input.lower().strip().split())
    
    # Create a list of dish names
    dish_names = data["Dish Name"].unique().tolist()
    
    # Find the best match using token sort ratio for better word order handling
    match_result = process.extractOne(
        clean_input,
        dish_names,
        scorer=fuzz.token_sort_ratio
    )
    
    if match_result is None:
        return None, "Could not find any matches for the dish name."
        
    best_match, score = match_result

    if score >= similarity_threshold:
        logger.info(f"Found match '{best_match}' for '{dish_name_input}' with score {score}.")
        predicted_values = data[data["Dish Name"] == best_match][_nutrient_cols].iloc[0]
        prediction_df = pd.DataFrame([predicted_values])
        prediction_df.insert(0, "Dish Name (Predicted)", f"{dish_name_input} (similar to: {best_match})")
        return prediction_df, f"Prediction based on the closest match: '{best_match}' (Similarity: {score}%)"
    else:
        logger.info(f"No confident match found for '{dish_name_input}'. Best attempt: '{best_match}' with score {score}.")
        return None, f"Could not find a confident match for '{dish_name_input}'. Best guess was '{best_match}' (Similarity: {score}%), but it's below the threshold of {similarity_threshold}%."

if st.button("Find Nutrition", key="predict_button") and user_dish_name:
    predicted_df, message = predict_nutrition_from_text(user_dish_name, nutrition_data, nutrient_types)
    st.info(message)
    if predicted_df is not None:
        # Display the predicted nutrition values, formatted
        display_df = predicted_df.copy()
        format_dict = {col: '{:.2f}' for col in nutrient_types}
        st.dataframe(
            display_df.style.format(format_dict),
            hide_index=True,
            use_container_width=True
        )

# After the nutrition prediction section
with st.sidebar:
    st.header("Compare Dishes")
    # Add meal selection dropdowns
    meal_name = st.selectbox("Select first dish:", options=nutrition_data["Dish Name"].unique(), key="meal_1")
    meal_name_2 = st.selectbox("Select second dish (optional):", 
                              options=["None"] + nutrition_data["Dish Name"].unique().tolist(),
                              key="meal_2")
    
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
    if meal_name_2 and meal_name_2 != meal_name: # Add second meal if selected and different
        meal_data_2 = nutrition_data[nutrition_data["Dish Name"] == meal_name_2].copy()
        comparison_df[f'{meal_name_2}'] = meal_data_2[nutrient_types].values[0]
    
    # Display the comparison
    st.subheader("Nutrition Values Comparison")
    st.dataframe(
        comparison_df.style.format({
            f'{meal_name}': '{:.2f}',
            f'{meal_name_2}': '{:.2f}',
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
    if meal_name_2 and meal_name_2 != meal_name: # Apply xOffset only if comparing two meals
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