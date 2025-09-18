import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path)
wine = wine.drop(columns=['Id']) ## Id isnt a nessescary statistic for any part

corr_matrix = wine.corr() ## create a correlation matrix
print(corr_matrix) ## print out correlation matrix
print(corr_matrix['quality'].sort_values(ascending=False)) ## sort the values for best correlation with quality

sns.heatmap(data=corr_matrix, annot=True, cmap='cividis', fmt=".2f") ## create heatmap
plt.show() # plot heatmap

