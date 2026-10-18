import pandas as pd
import os

# 1. Load the raw data
input_path = os.path.join("data", "raw", "Sleep_health_and_lifestyle_dataset.csv")
df = pd.read_csv(input_path)

# 2. Handle Target Variable (Fill NaN with 'None' and map to integers)
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
target_mapping = {'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2}
df['Target_Sleep_Disorder'] = df['Sleep Disorder'].map(target_mapping)
df = df.drop(columns=['Sleep Disorder'])

# 3. Handle Blood Pressure (Split string into two numeric columns)
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
df = df.drop(columns=['Blood Pressure'])

# 4. Standardize categorical inconsistencies
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

# 5. Drop identifiers that cause data leakage
df = df.drop(columns=['Person ID'])

# 6. Save the processed data for modeling
output_path = os.path.join("data", "processed", "sleep_cleansed.csv")
df.to_csv(output_path, index=False)

print("--- Preprocessing Complete ---")
print(f"File saved to: {output_path}")
print("\nNew Shape:", df.shape)
print("\nNew Columns & Data Types:")
print(df.dtypes)
print("\nTarget Distribution:")
print(df['Target_Sleep_Disorder'].value_counts())