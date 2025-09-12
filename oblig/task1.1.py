import pandas as pd
import matplotlib.pyplot as plt

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path) 
print(wine.head(5)) #first five rows
print(wine.info()) # information about the data
print(wine.drop(columns=['Id']).describe()) # summary statistics

wine.hist(bins=50, figsize=(12,8))
plt.show()
