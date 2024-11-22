import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset
iris_df = sns.load_dataset('iris')

# Scatter plot of petal length vs petal width, colored by species
sns.pairplot(iris_df, hue='species')
plt.suptitle('Pairwise Relationships in the Iris Dataset', y=1.02)
plt.show()
