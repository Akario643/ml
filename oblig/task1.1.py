import pandas as pd
import matplotlib.pyplot as plt

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path)
wine = wine.drop(columns=['Id']) ## Id isnt a nessescary statistic for any part
print(wine.head(5)) #first five rows
print(wine.info()) # information about the data
print(wine.describe()) # summary statistics

wine.hist(bins=50, figsize=(12,8)) ## histogram
plt.show() # show the plot / histogram
