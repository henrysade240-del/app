import streamlit as st
from recommender import RecipeRecommender
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="🍳",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0f5132;
}

/* All main text */
html, body, [class*="css"] {
    color: white !important;
}

/* Text area + input fields */
textarea, input {
    background-color: #145a32 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Increase slider thickness */
div[data-baseweb="slider"] [role="slider"] {
    box-shadow: 0 0 8px rgba(46, 204, 113, 0.6);
    border: 3px solid white !important;
    background-color: #198754 !important;
}

/* Active track */
div[data-baseweb="slider"] div[aria-valuenow] + div {
    background-color: #2ecc71 !important;
    height: 8px !important;
    border-radius: 8px !important;
}

/* Track background */
div[data-baseweb="slider"] div[aria-valuenow] {
    height: 8px !important;
    border-radius: 8px !important;
    background-color: #cce3d4 !important;
}

/* Make slider value white */
div[data-baseweb="slider"] span {
    color: white !important;
    font-weight: bold;
}



/* Button styling */
.stButton > button {
    background-color: white !important;
    color: #0f5132 !important;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}

/* Button hover */
.stButton > button:hover {
    background-color: #e6e6e6 !important;
    color: #0f5132 !important;
}

</style>
""", unsafe_allow_html=True)



# Initialize the recommender
@st.cache_resource
def load_recommender():
    return RecipeRecommender('recipes (1).csv')

recommender = load_recommender()

# App title and description
st.title("🍳 Ingredient-Based Recipe Recommendation System")
st.image("Food.jpg", use_container_width=True)
st.markdown("""
Welcome! Enter the ingredients you have, and I'll recommend the best recipes for you.
The system uses **TF-IDF vectorization** and **cosine similarity** to find the most relevant matches.
""")

# Sidebar for filters
st.sidebar.header("Dietary Filters")
show_all = st.sidebar.checkbox("Show all recipes", value=True)

if not show_all:
    filter_options = st.sidebar.multiselect(
        "Exclude allergens:",
        ["eggs", "dairy", "nuts", "soy"],
        default=[]
    )
else:
    filter_options = []

# Main input area
st.header("Enter Your Ingredients")
user_input = st.text_area(
    "Type ingredients separated by commas (e.g., chicken, rice, garlic, onion):",
    height=100,
    placeholder="chicken, garlic, onion, tomato"
)

# Number of recommendations
num_recipes = st.slider("Number of recipes to show:", min_value=1, max_value=20, value=5)

# Search button
if st.button("🔍 Find Recipes", type="primary"):
    if user_input.strip():
        with st.spinner("Searching for the best recipes..."):
            # Get recommendations
            recommendations = recommender.recommend(
                user_input,
                top_n=num_recipes,
                exclude_allergens=filter_options
            )
            
            if recommendations:
                st.success(f"Found {len(recommendations)} recipes for you!")
                
                # Display results
                for idx, recipe in enumerate(recommendations, 1):
                    with st.expander(f"#{idx} - {recipe['recipe']} (Match: {recipe['similarity']:.1%})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**Ingredients:**")
                            st.write(recipe['ingredients'])
                        
                        with col2:
                            allergens = recipe['allergens']
                            if allergens and allergens.lower() != 'none':
                                st.warning(f"⚠️ Contains: {allergens}")
                            else:
                                st.success("✅ No common allergens")
                        
                        # Similarity score visualization
                        st.progress(recipe['similarity'])
                        # ---- Nutritional Information ----
                        st.markdown("### 🥗 Nutritional Information")

                        col_n1, col_n2, col_n3, col_n4 = st.columns(4)

                        col_n1.metric("Calories", f"{recipe['calories']} kcal")
                        col_n2.metric("Protein", f"{recipe['protein']} g")
                        col_n3.metric("Carbs", f"{recipe['carbs']} g")
                        col_n4.metric("Fats", f"{recipe['fats']} g")

                        # ---- Macro Chart ----
                        macro_data = pd.DataFrame({
                            "Macronutrient": ["Protein", "Carbs", "Fats"],
                            "Grams": [
                                recipe['protein'],
                                recipe['carbs'],
                                recipe['fats']
                            ]
                        })

                        st.bar_chart(macro_data.set_index("Macronutrient"))

            else:
                st.warning("No recipes found. Try different ingredients or remove some filters.")
    else:
        st.error("Please enter at least one ingredient!")

# Footer with information
st.divider()
st.markdown("""
### How it works:
1. **TF-IDF Vectorization**: Converts ingredient lists into numerical vectors that capture the importance of each ingredient
2. **Cosine Similarity**: Measures how similar your ingredients are to each recipe (0 = no match, 1 = perfect match)
3. **Ranking**: Shows you the top matches based on similarity scores

**Dataset**: Contains over 130 recipes with allergen information (eggs, dairy, nuts, soy)
""")
