# src/data_cleaning.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    """Load the dataset from a specified file path.

    Args:
        file_path (str): The path to the CSV file containing the data.

    Returns:
        DataFrame: A pandas DataFrame containing the loaded data.
    """
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    """Clean the dataset by handling missing values, encoding categorical features, and normalizing numerical columns if necessary.

    Args:
        data (DataFrame): The raw data to be cleaned.

    Returns:
        DataFrame: A cleaned DataFrame with encoded categorical variables.
    """
    data = data.dropna()
    categorical_columns = ['Age_Group', 'Gender', 'Socioeconomic_Status', 'School_Programs', 'Access_to_Counseling',
                           'Substance_Education']
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data


def save_cleaned_data(data, output_path):
    """Save the cleaned data to the specified output path.

    Args:
        data (DataFrame): The cleaned data to save.
        output_path (str): The file path where the cleaned data will be saved.
    """
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    """Load, clean, and save the dataset."""
    input_file = "../data/youth_smoking_drug_data.csv"
    output_file = "../data/processed/youth_smoking_drug_data_cleaned.csv"
    data = load_data(input_file)
    cleaned_data = clean_data(data)
    save_cleaned_data(cleaned_data, output_file)
