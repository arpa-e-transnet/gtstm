import pandas as pd
july_data = pd.read_csv("july_data.csv")
jan_data = pd.read_csv("jan.csv")
july_links = set(july_data["linkid"].unique())
jan_links = set(jan_data["linkid"].unique())
print jan_links-july_links
