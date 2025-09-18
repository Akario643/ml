from helperFunctions import wine ## dataset
import matplotlib.pyplot as plt

print(wine.head()) ## print first five rows
print(wine.info()) ## information about the data
print(wine.describe()) ## summary statistics

wine.hist(bins=50, figsize=(12,8))
plt.show()