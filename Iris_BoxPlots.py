import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_df = sns.load_dataset('iris')

# Scatter plot of petal length vs petal width, colored by species
plt.figure(figsize=(8, 6))
sns.boxplot(x='species', y='sepal_width', data=iris_df)
plt.title('Sepal Width by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.show()
