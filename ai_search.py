import re
import textwrap

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

# Load the dataset
file_path = r"uae_real_estate_2024_updated.csv"
data = pd.read_csv(file_path)

# Normalize city and area data for case-insensitive matching
data["city"] = data["city"].str.capitalize()
data["area_1"] = data["area_1"].str.capitalize()

# Combine relevant fields for AI-based matching
data["combined_features"] = data[["title", "displayAddress", "description"]].fillna("").agg(" ".join, axis=1)

# Load the pre-trained spaCy language model
nlp = spacy.load("en_core_web_sm")


# Helper function to wrap text
def wrap_text(dataframe, column, width=50):
    """
    Wrap text in a specified column of a dataframe to the given width.
    """
    dataframe[column] = dataframe[column].apply(lambda x: "\n".join(textwrap.wrap(str(x), width)))
    return dataframe


def extract_preferences_from_prompt(prompt):
    """
    Use spaCy to extract keywords, price, and size from the user's prompt.
    """
    doc = nlp(prompt)

    # Extract nouns and adjectives as keywords
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ"]]

    # Extract price (e.g., "$5000000", "5 million")
    price_matches = re.findall(r"(\d[\d,]*\s?(million|k|thousand)?)", prompt, flags=re.IGNORECASE)
    if price_matches:
        price_string = price_matches[0][0]
        if "million" in price_string.lower():
            max_price = int(float(price_string.replace("million", "").replace(",", "").strip()) * 1_000_000)
        elif "k" in price_string.lower() or "thousand" in price_string.lower():
            max_price = int(
                float(price_string.replace("k", "").replace("thousand", "").replace(",", "").strip()) * 1_000
            )
        else:
            max_price = int(price_string.replace(",", "").strip())
    else:
        max_price = float("inf")  # Default to no price limit

    # Extract size (e.g., "1000 sqft", "1k sq ft")
    size_matches = re.search(r"(\d[\d,]*)\s?(sqft|sq ft|k)", prompt, flags=re.IGNORECASE)
    if size_matches:
        size_string = size_matches.group(1)
        min_size = int(size_string.replace(",", "").strip())
        if "k" in prompt.lower():
            min_size *= 1_000  # Convert 'k' to thousands
    else:
        min_size = 0  # Default to no size limit

    return keywords, max_price, min_size


def recommend_properties_ai(prompt, data, top_n=5):
    """
    Recommend properties based on AI-extracted preferences, max price, and min size.
    """
    # Extract preferences from the prompt
    keywords, max_price, min_size = extract_preferences_from_prompt(prompt)
    print(f"Extracted Keywords: {keywords}")
    print(f"Extracted Max Price: {max_price}")
    print(f"Extracted Min Size: {min_size}")

    # Convert keywords into a single string for TF-IDF vectorization
    keyword_string = " ".join(keywords)

    # Vectorize the dataset and keywords
    vectorizer = TfidfVectorizer(stop_words="english")
    feature_matrix = vectorizer.fit_transform(data["combined_features"])
    user_vector = vectorizer.transform([keyword_string])

    # Calculate similarity scores
    similarity_scores = cosine_similarity(user_vector, feature_matrix).flatten()

    # Filter by max price and min size
    filtered_data = data[(data["price"] <= max_price) & (data["sizeMin"] >= min_size)].copy()
    if filtered_data.empty:
        print("No properties match your filters. Try adjusting max price or size.")
        return pd.DataFrame()  # Return an empty DataFrame if no matches

    # Calculate similarity scores for filtered data
    filtered_similarity_scores = similarity_scores[filtered_data.index]

    # Get top N property indices
    top_indices = filtered_similarity_scores.argsort()[-top_n:][::-1]
    return filtered_data.iloc[top_indices]


# Display recommended properties in a clean tabular format
def display_properties(properties, title="Top Properties"):
    """
    Display properties in a tabular format.
    """
    if not properties.empty:
        print(f"\n{title}:")
        formatted_output = properties[["title", "displayAddress", "price", "sizeMin", "description"]].copy()
        formatted_output = wrap_text(formatted_output, "description", width=50)
        print(
            tabulate(
                formatted_output,
                headers=["Title", "Address", "Price", "Size (sqft)", "Description"],
                tablefmt="fancy_grid",
                showindex=False,
            )
        )
    else:
        print("No properties to display.")


# Get user prompt
user_prompt = input(
    "Describe the type of property you are looking for (e.g., modern villa, $2 million, 3 bedrooms, 2000 sqft):\n"
)

# Recommend properties using AI
recommended_properties = recommend_properties_ai(user_prompt, data, top_n=5)

# Display the recommendations
display_properties(recommended_properties, title="Top 5 Properties Based on Your Preferences")

if __name__ == "__main__":
    file_path = r"uae_real_estate_2024_updated.csv"
    data = pd.read_csv(file_path)
    user_prompt = input(
        "Describe the type of property you are looking for (e.g., modern villa, $2 million, 3 bedrooms, 2000 sqft):\n"
    )
    recommended_properties = recommend_properties_ai(user_prompt, data)

    print(recommended_properties)
