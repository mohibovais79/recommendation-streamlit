import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabulate import tabulate

# Load the dataset
file_path = r"uae_real_estate_2024_updated.csv"
data = pd.read_csv(file_path)

# Data Preprocessing
# Encode categorical columns
categorical_columns = ["city", "area_1", "furnishing"]
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])

# Handle numerical columns (convert to appropriate types if needed)
data["bedrooms"] = pd.to_numeric(data["bedrooms"], errors="coerce")
data["bathrooms"] = pd.to_numeric(data["bathrooms"], errors="coerce")
data["sizeMin"] = pd.to_numeric(data["sizeMin"], errors="coerce")

# Fill any missing numerical values with the mean (if any)
numerical_columns = ["bedrooms", "bathrooms", "price", "sizeMin"]
for col in numerical_columns:
    data[col] = data[col].fillna(data[col].mean())

# Combine text-based features into a single string for similarity-based recommendations
data["features"] = (
    data[["city", "area_1", "bedrooms", "bathrooms", "furnishing", "description"]]
    .astype(str)
    .fillna("")
    .agg(" ".join, axis=1)
)

# Recommendation Functionality
# Vectorize the textual features for similarity-based recommendations
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(data["features"])


def safe_transform(label_encoder, value):
    """
    Safely transform a value using a LabelEncoder.
    If the value is not seen during training, return the closest match or 'Other'.
    """
    # Capitalize input to match training data
    value = value.strip().capitalize()
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    else:
        # Add an "Other" class if not present in the encoder
        if "Other" not in label_encoder.classes_:
            label_encoder.classes_ = np.append(label_encoder.classes_, "Other")
        return label_encoder.transform(["Other"])[0]


def recommend_properties(user_input, data, feature_matrix, max_price, min_size, top_n=10):
    """
    Recommend properties based on user input using cosine similarity, max price, and size constraints.
    """
    # Filter based on price and size
    filtered_data = data[(data["price"] <= max_price) & (data["sizeMin"] >= min_size)]
    if filtered_data.empty:
        print("No properties match your filters. Try adjusting max price or size.")
        return pd.DataFrame()  # Return an empty DataFrame if no matches

    user_features = " ".join(user_input.values())
    user_vector = vectorizer.transform([user_features])
    similarity_scores = cosine_similarity(user_vector, feature_matrix[filtered_data.index])
    top_indices = similarity_scores.argsort()[0, -top_n:][::-1]
    return filtered_data.iloc[top_indices]


# Default values for user input
default_values = {
    "city": "Dubai",
    "area_1": "Downtown",
    "bedrooms": "3",
    "bathrooms": "2",
    "furnishing": "Furnished",
    "description": "Spacious and modern",
    "max_price": 5000000,
    "min_size": 1000,
}

# Example User Input with defaults
user_input_example = {
    "city": input(f"Enter city (default: {default_values['city']}): ") or default_values["city"],
    "area_1": input(f"Enter area (default: {default_values['area_1']}): ") or default_values["area_1"],
    "bedrooms": input(f"Enter number of bedrooms (default: {default_values['bedrooms']}): ")
    or default_values["bedrooms"],
    "bathrooms": input(f"Enter number of bathrooms (default: {default_values['bathrooms']}): ")
    or default_values["bathrooms"],
    "furnishing": input(
        f"Enter furnishing preference (Furnished/Unfurnished, default: {default_values['furnishing']}): "
    )
    or default_values["furnishing"],
    "description": input(f"Enter any additional preferences (default: {default_values['description']}): ")
    or default_values["description"],
}

# Get max price and min size inputs
max_price = float(
    input(f"Enter maximum price (default: {default_values['max_price']}): ") or default_values["max_price"]
)
min_size = float(
    input(f"Enter minimum size in sqft (default: {default_values['min_size']}): ") or default_values["min_size"]
)

# Transform user input into a format compatible with the recommendation system
user_input_encoded = {
    "city": safe_transform(label_encoders["city"], user_input_example["city"]),
    "area_1": safe_transform(label_encoders["area_1"], user_input_example["area_1"]),
    "bedrooms": float(user_input_example["bedrooms"]),
    "bathrooms": float(user_input_example["bathrooms"]),
    "furnishing": safe_transform(label_encoders["furnishing"], user_input_example["furnishing"]),
}

# Get Recommendations
recommended_properties = recommend_properties(user_input_example, data, feature_matrix, max_price, min_size)


# Helper function to wrap text
def wrap_text(dataframe, column, width=50):
    """
    Wrap text in a specified column of a dataframe to the given width.
    """
    dataframe[column] = dataframe[column].apply(lambda x: "\n".join(textwrap.wrap(x, width)))
    return dataframe


# Display top 10 recommended properties in a clean tabular format
if not recommended_properties.empty:
    print("\nTop 10 Recommended Properties:")
    # Wrap text in the 'description' column for readability
    formatted_output = recommended_properties[["title", "displayAddress", "description"]].copy()
    formatted_output = wrap_text(formatted_output, "description", width=50)
    print(
        tabulate(formatted_output, headers=["Title", "Address", "Description"], tablefmt="fancy_grid", showindex=False)
    )
else:
    print("No properties to display.")
