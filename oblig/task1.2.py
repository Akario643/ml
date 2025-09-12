import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'oblig\WineQT.csv' # you might have to change this to run locally
wine = pd.read_csv(file_path) 

corr_matrix = wine.drop(columns=['Id']).corr()
print(corr_matrix)
print(corr_matrix['quality'].sort_values(ascending=False))

sns.heatmap(data=corr_matrix, annot=True, cmap='cividis', fmt=".2f")
plt.show()

