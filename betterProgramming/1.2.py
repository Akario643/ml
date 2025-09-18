from helperFunctions import wine # import data
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = wine.corr() ## create correlation matrix
print(corr_matrix)
print(corr_matrix['quality'].sort_values(ascending=False)) ## sort the values for best correlation with quality

sns.heatmap(data=corr_matrix, annot=True, cmap='cividis', fmt=".2f") ## create heatmap
plt.show() # plot heatmap