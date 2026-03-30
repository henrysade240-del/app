
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RecipeRecommender:
    def __init__(self, csv_path):
        """
        Initialize the Recipe Recommender System
        
        Args:
            csv_path: Path to the recipes CSV file
        """
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.lower()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=500
        )
        
        # Fit the vectorizer on all recipe ingredients
        self.recipe_vectors = self.vectorizer.fit_transform(self.df['ingredients'])
        
    def recommend(self, user_ingredients, top_n=5, exclude_allergens=None):
        """
        Recommend recipes based on user ingredients
        
        Args:
            user_ingredients: String of comma-separated ingredients
            top_n: Number of recommendations to return
            exclude_allergens: List of allergens to exclude
            
        Returns:
            List of dictionaries containing recipe information
        """
        if exclude_allergens is None:
            exclude_allergens = []
        
        # Clean and prepare user input
        user_ingredients = user_ingredients.lower().strip()
        
        # Transform user ingredients to vector
        user_vector = self.vectorizer.transform([user_ingredients])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, self.recipe_vectors).flatten()
        
        # Add similarity scores to dataframe
        df_with_scores = self.df.copy()
        df_with_scores['similarity'] = similarities
        
        # Filter out allergens if specified
        if exclude_allergens:
            for allergen in exclude_allergens:
                df_with_scores = df_with_scores[
                    ~df_with_scores['allergens'].str.contains(allergen, case=False, na=False)
                ]
        
        # Sort by similarity and get top N
        top_recipes = df_with_scores.nlargest(top_n, 'similarity')
        
        # Format results
        results = []
        for _, row in top_recipes.iterrows():
           results.append({
    'recipe': row['recipe'],
    'ingredients': row['ingredients'],
    'allergens': row['allergens'],
    'similarity': row['similarity'],
    'calories': row['calories'],
    'protein': row['protein'],
    'carbs': row['carbs'],
    'fats': row['fats']
})
        return results
