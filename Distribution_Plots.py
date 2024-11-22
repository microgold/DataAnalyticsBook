# Distributions

import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# Create a histogram with KDE for the 'total_bill' column
sns.histplot(tips['total_bill'], kde=True, color='blue')

# Adjust layout to prevent the title from being cut off
plt.tight_layout()

# Add title and labels
plt.title('Distribution of Total Bill Amounts')
plt.xlabel('Total Bill ($)')
plt.ylabel('Frequency')
plt.show()


# Create a scatter plot with a regression line
sns.lmplot(x='total_bill', y='tip', data=tips)
# Add title
plt.title('Relationship Between Total Bill and Tip')
plt.show()


# Create a box plot to compare tips across different days
sns.boxplot(x='day', y='tip', data=tips)

# Add title
plt.title('Tip Amounts by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Tip Amount ($)')
plt.show()

# Creating a violin plot to visualize the distribution of tips by day of the week
sns.violinplot(x='day', y='tip', data=tips, inner='quartile', palette='coolwarm')

# Add title and labels
plt.title('Distribution of Tip Amounts by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Tip Amount ($)')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

