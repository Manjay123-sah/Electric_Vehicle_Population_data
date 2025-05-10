import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r"C:\Users\manja\Downloads\Electric_Vehicle_Population_Data.csv")


df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df['City'] = df.get('City', pd.Series(dtype=str)).astype(str).str.strip().str.title()
df['State'] = df.get('State', pd.Series(dtype=str)).astype(str).str.strip().str.title()
df['Make'] = df.get('Make', pd.Series(dtype=str)).astype(str).str.strip().str.title()
df['Model'] = df.get('Model', pd.Series(dtype=str)).astype(str).str.strip().str.title()
df['Electric Vehicle Type'] = df.get('Electric Vehicle Type', pd.Series(dtype=str)).astype(str).str.strip().str.title()
df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')
df['Base MSRP'] = pd.to_numeric(df['Base MSRP'], errors='coerce')

print("Columns in dataset:", df.columns.tolist())


print("\nFirst 10 Rows")
print(df.head(5))

print("\nLast 10 Rows")
print(df.tail(5))

# Plot 1: Most Common EV Makes
top_makes = df['Make'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_makes_df = pd.DataFrame({
    'Make': top_makes.index,
    'Count': top_makes.values
})
sns.barplot(data=top_makes_df, x='Count', y='Make', hue='Make', palette='coolwarm', legend=False)
plt.title('Top 10 EV Makes')
plt.xlabel('Count')
plt.ylabel('Make')
plt.tight_layout()
plt.show()

# Plot 2: Electric Range Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Electric Range'].dropna(), bins=30, kde=True, color='teal')
plt.title('Distribution of Electric Vehicle Ranges')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Plot 3: EVs Over Model Year
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Model Year', hue='Model Year', 
              order=sorted(df['Model Year'].dropna().unique()), palette='magma', legend=False)
plt.title('EV Count by Model Year')
plt.xticks(rotation=45)
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.tight_layout()
plt.show()

#Plot 4: Correlation Heatmap
numeric_cols = ['Electric Range', 'Base MSRP']
df_numeric = df[numeric_cols].dropna()
corr_matrix = df_numeric.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation between Electric Range and Base MSRP')
plt.tight_layout()
plt.show()

#Plot 5: Scatter Plot Model Year vs Base MSRP
df_clean = df[['Model Year', 'Base MSRP']].dropna()
df_clean['Model Year'] = df_clean['Model Year'].astype(int)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='Model Year', y='Base MSRP', color='darkorange', alpha=0.6, edgecolor='w')
plt.title('Scatter Plot: Model Year vs Base MSRP')
plt.xlabel('Model Year')
plt.ylabel('Base MSRP ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 6: Box Plot - Electric Range by EV Type 
filtered_df = df[df['Electric Vehicle Type'].isin(['Battery Electric Vehicle (Bev)', 'Plug-In Hybrid Electric Vehicle (Phev)'])]
filtered_df = filtered_df[['Electric Vehicle Type', 'Electric Range']].dropna()
print("\nFiltered Data for Box Plot:")
print(filtered_df.head())
print("Unique EV Types:", filtered_df['Electric Vehicle Type'].unique())
plt.figure(figsize=(8, 6))
sns.boxplot(data=filtered_df, x='Electric Vehicle Type', y='Electric Range', hue='Electric Vehicle Type', palette='Set2')
plt.title('Electric Range by Electric Vehicle Type')
plt.xlabel('Electric Vehicle Type')
plt.ylabel('Electric Range (miles)')
plt.tight_layout()
plt.show()

# Plot 7: Pie Chart - EV Type Distribution
df['Electric Vehicle Type'] = df['Electric Vehicle Type'].str.title().str.strip()
df['Electric Vehicle Type'] = df['Electric Vehicle Type'].replace({
    'Battery Electric Vehicle (Bev)': 'BEV',
    'Plug-In Hybrid Electric Vehicle (Phev)': 'PHEV'
})
ev_type_counts = df['Electric Vehicle Type'].value_counts()
print("EV Type Counts:\n", ev_type_counts)
plt.figure(figsize=(4, 4))
colors = ['mediumseagreen', 'skyblue']
plt.pie(ev_type_counts, labels=ev_type_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Distribution of Electric Vehicle Types')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Plot 8: Line Chart - EV Count Trend by Model Year 
model_year_counts = df['Model Year'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=model_year_counts.index, y=model_year_counts.values, marker='o', color='purple')
plt.title('Trend of Electric Vehicle Count by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
