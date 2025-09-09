# Group 5 Project

## Project Title: Classification of Government Gazette entries by type

### Problem Statement
The Government publishes official notices; these notices span multiple domains such as legislation, appointments, notices, tenders, and regulations.  
When published, these documents are not placed in categories. Classifying them could be time-consuming when done manually.  
Our goal here is to create a system that classifies these documents according to the domain they belong to.  
The system model is aimed at correctly predicting the domain.

---

## General Objectives
Our project is to build a machine learning system that takes Gazette entries and gives the domain to which the document belongs.

---

## Business Objectives
1. **Improve efficiency** — reduce errors in categorising government Gazettes when done manually.  
2. **Improve accessibility** — make it easier to locate specific Gazettes.  
3. **Improve analysis** — enable better tracking of government activities through publication trends.  

---

## Data Mining Goals
To achieve our business objectives, we will focus on the following goals:

1. **Build a Supervised Classification Model**  
   Develop a model that can automatically assign each Gazette entry to one of several predefined categories such as legislation, appointments, tenders, notices and regulations.  

2. **Automate Categorization**  
   Train and deploy the classifier to a production environment where it can handle new, incoming Gazette entries without human intervention.  

---

## Initial Project Success Criteria
We will consider the project a success if it meets the following criteria:  

- **Model Performance**: The classification model must achieve a minimum of 80% accuracy.  
- **Balanced Performance**: The model must demonstrate balanced precision and recall across all predefined categories.  

---

## Data Understanding

```python
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Mount Google Drive to access our datasets
from google.colab import drive
drive.mount('/content/drive')

# Define dataset path
var_path = "/content/drive/My Drive/Colab Notebooks/gazette_notices.csv"

# Load dataset
var_df_gazettes = pd.read_csv(var_path)

# Display the first 10 observations
var_df_gazettes.head(10)

# Display dataframe summary: columns, datatypes, and non-null counts
var_df_gazettes.info()

# Summary statistics of dataset
var_df_gazettes.describe()

# Show number of rows and columns
var_df_gazettes.shape

import pandas as pd
import matplotlib.pyplot as plt

# Clean data by removing rows with missing 'Act'
df = var_df_gazettes
df_clean = df.dropna(subset=['Act'])

# Get top 20 most frequent Acts
top_counts = df_clean['Act'].value_counts().head(20)

# Plot bar chart
plt.figure(figsize=(10, 6))
top_counts.plot(kind='barh', color='green')
plt.xlabel('Number of Notices')
plt.ylabel('Act')
plt.title('Top 20 Notices by Act')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Print summary
print("\nTop 20 Counts of Notices by Act:")
print(top_counts)
```
# **Data Preprocessing**
In this step, we extract only the columns that are most relevant for our analysis:  
- **Notice_No**: serves as a unique identifier for each record  
- **Details**: contains the main textual information that will be used as the input feature for modeling  
- **Act**: provides the legal reference associated with each notice  

We create a new dataframe with these features to reduce noise and focus on the information that matters for the modeling phase.  
Displaying the first 10 rows allows us to quickly verify that the correct columns have been selected. 

```python
# select the columns to use
var_selectedcolumns = ['Notice_No', 'Details', 'Act']

# Create a new dataframe with those columns
var_selected_df = var_df_gazettes[var_selectedcolumns].copy()

# Display the first 10 rows
var_selected_df.head(10)
```
### **2.Data Pre-processing (Cleaning)**
#### **2.1 Handling Missing Values**
We first check for missing values across the selected features.  

- **Act**:Rows with missing values are dropped to preserve data integrity.  
- **Details**: As the primary input feature, rows missing this information are also dropped, since they would not contribute meaningful input to the model.

```python
# Check missing values before cleaning
print('Missing values before cleaning')
print(var_selected_df.isnull().sum())

# Drop rows with missing 'Act'
var_selected_df.dropna(subset=['Act'], inplace=True)

# Drop rows with missing 'Details'
var_selected_df = var_selected_df.dropna(subset=['Details'])

# Check missing values after cleaning
print('Missing values After cleaning')
print(var_selected_df.isnull().sum())

var_selected_df.shape
```
#### **2.2 Handling Duplicate Values**

Before cleaning, we check for duplicate entries in the **`Notice_No`** column, which serves as a unique identifier for each record.  

- Duplicate rows are dropped, keeping only the first occurrence.  
- This ensures that each notice is represented once, preventing data redundancy and bias in the modeling phase.  

After dropping duplicates, we recheck to confirm that no duplicates remain and verify the final shape of the cleaned dataset.

---

```python
# Count duplicates in 'Notice_No' before cleaning
var_duplicates = var_selected_df.duplicated(subset=['Notice_No']).sum()
print(f"Number of duplicates in Notice_No before cleaning: {var_duplicates}")

# Drop duplicates and keep the first occurrence
var_processed_df = var_selected_df.drop_duplicates(subset=['Notice_No'], keep="first")
print(f"Shape after handling duplicates in Notice_No: {var_processed_df.shape}")

# Re-check duplicates in the processed dataframe
var_duplicates = var_processed_df.duplicated(subset=['Notice_No']).sum()
print(f"Number of duplicates in Notice_No after cleaning: {var_duplicates}")

# Show the final shape of the cleaned dataframe
print("Final dataframe shape:", var_processed_df.shape)
