import streamlit as st
from datetime import timedelta
import pandas as pd
import altair as alt
import logging

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
    logger.info("Data loaded successfully")
    st.dataframe(nutrition_data, hide_index=True, height=300)

# Calculate average nutrition values
nutrient_types = nutrition_data.columns[1:]
average_nutrition = pd.DataFrame({
    'Dish Name': ['Average'],
    **{col: [nutrition_data[col].mean()] for col in nutrient_types}
})

with st.sidebar:
    st.header("Compare Meals")
    meal_name = st.selectbox("Choose first meal", nutrition_data["Dish Name"].unique(), key="meal_name")
    meal_name_2 = st.selectbox("Choose second meal", nutrition_data["Dish Name"].unique(), key="meal_name_2")

if meal_name:
    # Get selected meal data
    meal_data = nutrition_data[nutrition_data["Dish Name"] == meal_name].copy()
    meal_data_2 = nutrition_data[nutrition_data["Dish Name"] == meal_name_2].copy()
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Nutrient': nutrient_types,
        f'{meal_name}': meal_data[nutrient_types].values[0],
        f'{meal_name_2}': meal_data_2[nutrient_types].values[0],
        'Average': [nutrition_data[col].mean() for col in nutrient_types]
    })
    
    # Display the comparison
    st.subheader("Nutrition Values Comparison")
    st.dataframe(
        comparison_df.style.format({
            f'{meal_name}': '{:.2f}',
            f'{meal_name_2}': '{:.2f}',
            'Average': '{:.2f}'
        }),
        hide_index=True,
        use_container_width=True
    )
    
    # Calculate normalized values for both meals
    for nutrient in nutrient_types:
        avg_value = nutrition_data[nutrient].mean()
        meal_data[nutrient] = meal_data[nutrient] / avg_value
        meal_data_2[nutrient] = meal_data_2[nutrient] / avg_value
    
    # Create bar chart data for both meals
    chart_data = pd.concat([
        meal_data.melt(id_vars=['Dish Name'], var_name='Nutrient', value_name='Value').assign(Meal=meal_name),
        meal_data_2.melt(id_vars=['Dish Name'], var_name='Nutrient', value_name='Value').assign(Meal=meal_name_2)
    ])
    
    # Create bar chart
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Nutrient:N', title='Nutrient'),
        xOffset='Meal:N',  # This creates the side-by-side grouping
        y=alt.Y('Value:Q', 
                title='Value (Relative to Average)',
                axis=alt.Axis(format='%')),
        color=alt.Color('Meal:N', title='Meal'),
        tooltip=['Nutrient', 'Meal', alt.Tooltip('Value:Q', format='.2%')]
    ).properties(
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