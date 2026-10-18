import pandas as pd
import os

# List of the files you downloaded
files = ["DEMO_J.XPT", "BMX_J.XPT", "SLQ_J.XPT", "DIQ_J.XPT", "BPQ_J.XPT"]
df_merged = None

print("Merging local CDC NHANES datasets...")

for file in files:
    file_path = os.path.join("data", "raw", file)
    print(f"Reading {file}...")
    try:
        # read_sas can read the local XPT files directly
        df_temp = pd.read_sas(file_path)
        if df_merged is None:
            df_merged = df_temp
        else:
            # Merge on SEQN (Respondent sequence number)
            df_merged = pd.merge(df_merged, df_temp, on='SEQN', how='inner')
            
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please ensure it is in the data\\raw folder.")
    except Exception as e:
        print(f"Error processing {file}: {e}")

if df_merged is not None and not df_merged.empty:
    output_path = os.path.join("data", "raw", "nhanes_raw.csv")
    df_merged.to_csv(output_path, index=False)
    
    print("\n--- NHANES Local Merge Complete ---")
    print(f"File saved to: {output_path}")
    print(f"Total Participants (Rows): {df_merged.shape[0]}")
    print(f"Total Variables (Columns): {df_merged.shape[1]}")