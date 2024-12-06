import textwrap

import pandas as pd
from tabulate import tabulate

# Load the dataset
file_path = r"uae_real_estate_2024_updated.csv"
data = pd.read_csv(file_path)

# Normalize city and area data for case-insensitive matching
data["city"] = data["city"].str.capitalize()
data["area_1"] = data["area_1"].str.capitalize()


# Helper function to wrap text
def wrap_text(dataframe, column, width=50):
    """
    Wrap text in a specified column of a dataframe to the given width.
    """
    dataframe[column] = dataframe[column].apply(lambda x: "\n".join(textwrap.wrap(str(x), width)))
    return dataframe


def search_by_city_or_area(search_input, data=data, top_n=5):
    """
    Search for properties by city or area and recommend the top N properties.
    """
    # Normalize input for case-insensitive matching
    search_input = search_input.strip().capitalize()

    # Filter by city or area
    filtered_data = data[(data["city"] == search_input) | (data["area_1"] == search_input)]
    if filtered_data.empty:
        print(f"No properties found for '{search_input}'. Please try another city or area.")
        return pd.DataFrame()  # Return an empty DataFrame if no matches

    # Recommend top N properties
    return filtered_data.head(top_n)


# Display recommended properties in a clean tabular format
def display_properties(properties, title="Top Properties"):
    """
    Display properties in a tabular format.
    """
    if not properties.empty:
        print(f"\n{title}:")
        formatted_output = properties[["title", "displayAddress", "description"]].copy()
        formatted_output = wrap_text(formatted_output, "description", width=50)
        print(
            tabulate(
                formatted_output, headers=["Title", "Address", "Description"], tablefmt="fancy_grid", showindex=False
            )
        )
    else:
        print("No properties to display.")


# Search input from the user
search_input = input("Enter the city or area to search: ").strip()
search_results = search_by_city_or_area(search_input, data, top_n=5)
display_properties(search_results, title=f"Top 5 Properties in {search_input}")

if __name__ == "__main__":
    pass
