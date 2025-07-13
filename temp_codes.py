import pandas as pd

# 1. Read the Excel file into a DataFrame
df = pd.read_excel(r'C:\Users\btulu\OneDrive - University of Florida\Publication\koke\data\CGPP_Ethi_2020 to 2024 Suspected Measels cases.xlsx')  # supports .xlsx, .xls, etc. :contentReference[oaicite:0]{index=0}

# 2. Write the DataFrame out to CSV (no index column)
df.to_csv('Measels_Data.csv', index=False)  # writes comma-separated values :contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}
