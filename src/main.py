import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

file_path = '/kaggle/input/youth-smoking-and-drug-dataset/youth_smoking_drug_data_10000_rows_expanded.csv'
df = pd.read_csv(file_path)
df.head(10)
df = pd.read_csv(file_path)
df.head(10)

categorical_columns = ['Age_Group', 'Gender', 'Socioeconomic_Status', 'School_Programs', 'Access_to_Counseling', 'Substance_Education']
for col in categorical_columns:
    print(f"{col}:\n{df[col].value_counts()}\n")