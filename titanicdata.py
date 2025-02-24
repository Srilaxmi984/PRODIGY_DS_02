import pandas as pd

# Load the Titanic dataset (ensure you have downloaded it)
df = pd.read_csv('titanic.csv')

# Check the first few rows of the dataset
df.head()

# Check for missing values
df.isnull().sum()

# Handle missing values
# For 'Age', we might use the median or a different strategy
df['Age'].fillna(df['Age'].median(), inplace=True)

# For 'Embarked', we can fill with the mode (most frequent value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column as it has too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Check for any other missing values after imputation
df.isnull().sum()

# Convert 'Sex' column to a numerical format
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' column to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Final check of the cleaned data
df.head()
import matplotlib.pyplot as plt
import seaborn as sns

# Summary statistics
df.describe()

# Visualizing the distribution of 'Age' and 'Fare'
plt.figure(figsize=(12, 6))

# Age Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Age')

# Fare Distribution
plt.subplot(1, 2, 2)
sns.histplot(df['Fare'], bins=30, kde=True, color='green')
plt.title('Distribution of Fare')

plt.tight_layout()
plt.show()

# Countplot of 'Survived' by 'Sex'
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Sex')
plt.show()

# Countplot of 'Survived' by 'Pclass'
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Count by Pclass')
plt.show()

# Heatmap to show correlations between numeric features
plt.figure(figsize=(10, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
# Relationship between 'Survived' and 'Sex'
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Rate by Sex')
plt.show()

# Relationship between 'Survived' and 'Pclass'
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Rate by Pclass')
plt.show()

# Age vs Survived
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Survival Rate by Age')
plt.show()

# Fare vs Survived
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Survival Rate by Fare')
plt.show()
