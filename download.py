import pandas as pd

df = pd.read_csv("mental_health_analysis.csv")

df = df.set_index("User_ID")

df['Gender'] = df['Gender'].map({'F': 0,'M': 1})
df['Support_System'] = df['Support_System'].map({'Low': 0,'Moderate': 1, 'High': 2})
df['Academic_Performance'] = df['Academic_Performance'].map({'Poor': 0,'Average': 1, 'Good': 2, 'Excellent': 3})

df.to_csv('df_clear.csv')




