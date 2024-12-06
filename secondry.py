import textwrap

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from ai_search import recommend_properties_ai
from search_by_city import search_by_city_or_area

# Load the dataset
file_path = r"uae_real_estate_2024_updated.csv"
data = pd.read_csv(file_path)

# Normalize city and area data for case-insensitive matching
data["city"] = data["city"].str.capitalize()
data["area_1"] = data["area_1"].str.capitalize()

# Combine relevant fields for AI-based matching
data["combined_features"] = data[["title", "displayAddress", "description"]].fillna("").agg(" ".join, axis=1)

# Data Preprocessing
categorical_columns = ["city", "area_1", "furnishing"]
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])

data["bedrooms"] = pd.to_numeric(data["bedrooms"], errors="coerce")
data["bathrooms"] = pd.to_numeric(data["bathrooms"], errors="coerce")
data["sizeMin"] = pd.to_numeric(data["sizeMin"], errors="coerce")

numerical_columns = ["bedrooms", "bathrooms", "price", "sizeMin"]
for col in numerical_columns:
    data[col] = data[col].fillna(data[col].mean())

data["features"] = (
    data[["city", "area_1", "bedrooms", "bathrooms", "furnishing", "description"]]
    .astype(str)
    .fillna("")
    .agg(" ".join, axis=1)
)

vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(data["features"])


def safe_transform(label_encoder, value):
    value = value.strip().capitalize()
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    else:
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
        return pd.DataFrame()  # Return an empty DataFrame if no matches

    # Convert all user input values to strings for joining
    user_features = " ".join(str(value) for value in user_input.values())
    user_vector = vectorizer.transform([user_features])
    similarity_scores = cosine_similarity(user_vector, feature_matrix[filtered_data.index])
    top_indices = similarity_scores.argsort()[0, -top_n:][::-1]
    return filtered_data.iloc[top_indices]


def wrap_text(dataframe, column, width=50):
    dataframe[column] = dataframe[column].apply(lambda x: "\n".join(textwrap.wrap(x, width)))
    return dataframe


# Main Title Page
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>AI Real Estate Recommendation</h1>", unsafe_allow_html=True
    )
    st.video("videoplayback.mp4", format="video/mp4", autoplay=True, loop=True, muted=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

    # New AI Search Prompt
    ai_search_prompt = st.text_input("Enter your search prompt (e.g., '3 bedroom apartment in Dubai')")

    if st.button("Search AI"):
        recommended_properties_prompt = recommend_properties_ai(ai_search_prompt, data)
        if not recommended_properties_prompt.empty:
            st.subheader("Top Recommended Properties from AI Search:")
            formatted_output_prompt = recommended_properties_prompt[
                ["title", "displayAddress", "description", "price"]
            ].copy()
            formatted_output_prompt = wrap_text(formatted_output_prompt, "description", width=50)
            st.table(formatted_output_prompt)
        else:
            st.warning("No properties match your prompt. Try adjusting your search terms.")

    # Button to go to Filtering Page
    if st.button("Go to Filtering Options"):
        st.session_state.page = "filtering"  # Change the page state
        st.markdown("</div>", unsafe_allow_html=True)
        st.spinner("Loading...")  # Show loading spinner
        st.rerun()
        # No need for experimental_rerun, Streamlit will re-render automatically

    # New Button to go to Search by City or Area Page
    if st.button("Search by City or Area"):
        st.session_state.page = "search_by_city"  # Change the page state
        st.markdown("</div>", unsafe_allow_html=True)
        st.spinner("Loading...")  # Show loading spinner
        st.rerun()

# Recommendation Page
if st.session_state.page == "recommendation":
    st.title("Property Recommendation System")

    # New Section for Prompt-Based Search
    st.subheader("Search by Prompt")
    prompt = st.text_input("Enter your search prompt (e.g., '3 bedroom apartment in Dubai')")

    # Button to get recommendations based on prompt
    if st.button("Get Recommendations by Prompt"):
        recommended_properties_prompt = recommend_properties_ai(prompt, data)

        if not recommended_properties_prompt.empty:
            st.subheader("Top Recommended Properties from Prompt:")
            formatted_output_prompt = recommended_properties_prompt[
                ["title", "displayAddress", "description", "price"]
            ].copy()
            formatted_output_prompt = wrap_text(formatted_output_prompt, "description", width=50)
            st.table(formatted_output_prompt)
        else:
            st.warning("No properties match your prompt. Try adjusting your search terms.")

# New Search by City or Area Page
if st.session_state.page == "search_by_city":
    st.title("Search Properties by City or Area")
    city_or_area = st.text_input("Enter city or area to search for properties:")

    if st.button("Search"):
        results = search_by_city_or_area(city_or_area)
        if not results.empty:
            st.subheader("Properties Found:")
            formatted_output = results[["title", "displayAddress", "description", "price"]].copy()
            formatted_output = wrap_text(formatted_output, "description", width=50)
            st.table(formatted_output)
        else:
            st.warning("No properties found for the specified city or area.")

# Filtering Page
if st.session_state.page == "filtering":
    st.title("Property Filtering Options")
    # User Inputs for Field-Based Search
    city = st.text_input("Enter city (default: Dubai)", "Dubai")
    area_1 = st.text_input("Enter area (default: Downtown)", "Downtown")
    bedrooms = st.number_input("Enter number of bedrooms (default: 3)", min_value=1, value=3)
    bathrooms = st.number_input("Enter number of bathrooms (default: 2)", min_value=1, value=2)
    furnishing = st.selectbox("Enter furnishing preference", ["Furnished", "Unfurnished"], index=0)
    description = st.text_input(
        "Enter any additional preferences (default: Spacious and modern)", "Spacious and modern"
    )
    max_price = st.number_input("Enter maximum price (default: 5000000)", min_value=0.0, value=5000000.0)
    min_size = st.number_input("Enter minimum size in sqft (default: 1000)", min_value=0.0, value=1000.0)

    # Button to get recommendations based on field inputs
    if st.button("Get Recommendations"):
        user_input_encoded = {
            "city": safe_transform(label_encoders["city"], city),
            "area_1": safe_transform(label_encoders["area_1"], area_1),
            "bedrooms": float(bedrooms),
            "bathrooms": float(bathrooms),
            "furnishing": safe_transform(label_encoders["furnishing"], furnishing),
        }

        recommended_properties = recommend_properties(user_input_encoded, data, feature_matrix, max_price, min_size)

        if not recommended_properties.empty:
            st.subheader("Top Recommended Properties:")
            formatted_output = recommended_properties[["title", "displayAddress", "description", "price"]].copy()
            formatted_output = wrap_text(formatted_output, "description", width=50)
            st.table(formatted_output)
        else:
            st.warning("No properties match your filters. Try adjusting max price or size.")
