import pandas as pd
import os

file_path = os.path.join("data", "raw", "large_sleep_data.csv")
df = pd.read_csv(file_path)
print(df.columns)