# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(file_path):
    """Preprocess the dataset by applying normalization to continuous features and one-hot encoding to categorical features,
    rounding numerical results to three decimal places.

    Args:
        file_path (str): Path to the cleaned CSV file.

    Returns:
        DataFrame: The preprocessed DataFrame with normalized continuous features and one-hot encoded categorical features.
    """
    data = pd.read_csv(file_path)

    # Continuous features including the new ones 'Year', 'Age_Group', and 'Parental_Supervision'
    continuous_features = ['Smoking_Prevalence', 'Drug_Experimentation', 'Peer_Influence',
                           'Family_Background', 'Mental_Health', 'Community_Support', 'Media_Influence',
                           'Year', 'Parental_Supervision']

    scaler = MinMaxScaler()

    # Categorical features
    categorical_features = ['Gender', 'Socioeconomic_Status']

    # One-hot encoder for categorical features
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')

    # Label encoding 'Age_Group'
    data['Age_Group'] = LabelEncoder().fit_transform(data['Age_Group'])

    # Binary features (No transformation needed for these)
    binary_features = ['School_Programs', 'Access_to_Counseling', 'Substance_Education']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', scaler, continuous_features),
            ('onehot', one_hot_encoder, categorical_features)
        ],
        remainder='passthrough'  # Keep other features (like binary) unchanged
    )

    # Apply transformations
    data_transformed = preprocessor.fit_transform(data)

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Create DataFrame with the preprocessed data
    data_preprocessed = pd.DataFrame(data_transformed, columns=feature_names).round(3)

    return data_preprocessed


if __name__ == "__main__":
    """Load, preprocess, and save the dataset with normalization and one-hot encoding applied to appropriate features,
    rounding the results to three decimal places."""
    input_file = "../data/processed/youth_smoking_drug_data_cleaned.csv"
    output_file = "../data/processed/youth_smoking_drug_data_final_preprocessed.csv"

    preprocessed_data = preprocess_data(input_file)

    preprocessed_data.to_csv(output_file, index=False)
    print("Data preprocessing completed and saved to final file.")
