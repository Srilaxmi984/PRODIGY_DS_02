### Task 02: Titanic Data Analysis in Python


#### 📄 Description


For this task, I was assigned to analyze the Titanic dataset using Python. The objective was to explore passenger survival patterns by examining various factors such as age, fare, gender, and passenger class.

#### 📊 Dataset


Sample Dataset: Titanic CSV file (commonly available on Kaggle).


Key Columns:


Survived: Indicates whether a passenger survived (1) or not (0).


Pclass: Passenger class (1st, 2nd, or 3rd).


Sex: Gender of the passenger.


Age: Age of the passenger.


Fare: Ticket fare paid.


Embarked: Port of embarkation.


#### 📌 Categorical Data Selection

I focused on several categorical variables:

Sex (male/female)

Pclass (1, 2, 3)

Survived (0, 1)

These columns allowed me to compare survival rates across different demographic groups.


#### 🔍 Data Analysis in Python


Data Import:


Loaded the Titanic CSV file into a pandas DataFrame.
Data Cleaning:


Checked for missing values using df.isnull().sum().

Filled missing ages with the median (df['Age'].median()).

Filled missing embarkation ports with the mode.

Dropped the Cabin column due to a large proportion of missing data.


Data Transformation:


Converted Sex into numeric format (0 for male, 1 for female).
Used one-hot encoding on the Embarked column to handle categorical values.


Statistical Summary:


Generated summary statistics using df.describe() to understand the basic distribution of features.


#### 🛠 Steps to Create the Visualizations


Age and Fare Distributions:

Plotted histograms with sns.histplot() for both Age and Fare, using 30 bins and a KDE curve to visualize the density.
Survival Count by Sex and Class:


Created count plots (sns.countplot()) to see how many survived based on Sex and Pclass.
Correlation Heatmap:


Constructed a heatmap using sns.heatmap() to highlight relationships between numeric features like Age, Fare, Survived, etc.
Boxplots for Age and Fare vs. Survival:


Used sns.boxplot() to compare the distributions of Age and Fare between those who survived and those who did not.
Chart Customization:

Assigned meaningful titles and axis labels.


Adjusted color schemes for clarity.


#### 📊 Output


Histograms: Display the distribution of Age and Fare, providing insights into the typical passenger profile.


Count Plots: Show distinct survival patterns by Sex and Pclass, highlighting higher survival rates for certain groups.


Heatmap: Illustrates how numerical variables correlate, helping identify the most influential features on survival.


Boxplots: Compare Age and Fare between survivors and non-survivors, revealing notable differences in these features.


#### 📈 Impact


Understanding Survival Patterns: The visualizations indicate that gender, class, and fare strongly influence survival rates.


Decision Support: These insights can guide further predictive modeling or more in-depth statistical analysis.


Data Skills Demonstration: Showcases proficiency in handling missing data, applying transformations, and generating clear, informative plots in Python.
