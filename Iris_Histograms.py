import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the Iris dataset
iris_df = sns.load_dataset('iris')

# Display the first few rows of the dataset
iris_df.head()

# Create histograms for sepal length and petal length
plt.figure(figsize=(10, 5))

# Sepal Length
plt.subplot(1, 2, 1)
sns.histplot(iris_df['sepal_length'], kde=True, color='blue')
plt.title('Distribution of Sepal Length')

# Petal Length
plt.subplot(1, 2, 2)
sns.histplot(iris_df['petal_length'], kde=True, color='green')
plt.title('Distribution of Petal Length')

plt.tight_layout()
plt.show()
