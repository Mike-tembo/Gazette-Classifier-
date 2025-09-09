Group 5 Project

# Project Title: Classification of Government Gazette entries by type

## Problem Statement:
The Government publishes official notices; these notices span multiple domains , such as legislation, appointments, notices, tenders, and regulations.When published these documents are not placed in categories. Classifying them could be time-consuming when done manually . Our goal here is to create a system that classifies these documents according to the domain they belong to.the system model is aimed at correctly predicting the domain.

## General Objectives

Our project is to build a machine learning system that takes  Gazette entries and gives the domain to which the document belongs.

## Business Objectives
1. Improve efficiency- reduce errors in categorising government Gazettes experienced when done manually.

2. Improve Government Gazette accessibility by making it easier to locate specific Gazettes.

3. Improve analysis of publication trends to better track government activities.

## Data Mininig Goals

To achieve our business objectives, we will focus on the following data mining goals:

1. Build a Supervised Classification Model: We will develop a model that can
automatically assign each Gazette entry to one of several predefined categories such as
legislation, appointments, tenders, notices and regulations

2. Automate Categorization: Train and deploy the classifier to a production
environment where it can handle new, incoming Gazette entries without human
intervention.

## Initial Project Success Criteria

We will consider the project a success if it meets the following criteria:
Model Performance: The classification model must achieve a minimum of 80% accuracy.
Balanced Performance: The model must demonstrate balanced precision and recall across all of the predefined categories.


## Data Understanding
import pandas as pd
import matplotlib.pyplot as plt

### Mount Google Drive to access our datasets
from google.colab import drive
drive.mount('/content/drive')

var_path ="/content/drive/My Drive/Colab Notebooks/gazette_notices.csv"
var_df_gazettes =pd.read_csv(var_path)

### PERFORM BASIC OPERATIONS TO UNDERSTAND THE DATA
Displaying the first 5 observation of the dataset
var_df_gazettes.head(10)

### Summary of the dataFrame ,displaying the number of columns and rows, and datatype with non-null values
var_df_gazettes.info()

### Summary of the structure of dataset
var_df_gazettes.describe()

### Displaying the number of rows and columns in the dataset
var_df_gazettes.shape

### visualization of the dataset
import pandas as pd
import matplotlib.pyplot as plt


df = var_df_gazettes
df_clean = df.dropna(subset=['Act'])
top_counts = df_clean['Act'].value_counts().head(20)

plt.figure(figsize=(10, 6))
top_counts.plot(kind='barh', color='green')
plt.xlabel('Number of Notices')
plt.ylabel('Act')
plt.title('Top 20 Notices by Act')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


print("\nTop 20 Counts of Notices by Act:")
print(top_counts)

# **Data Preprocessing**
### **1. Data Selection**
In this step, we extract only the columns that are most relevant for our analysis:

Notice_No: serves as a unique identifier for each record
Details: contains the main textual information that will be used as the input feature for modeling
Act: provides the legal reference associated with each notice
We create a new dataframe with these features to reduce noise and focus on the information that matters for the modeling phase.
Displaying the first 10 rows allows us to quickly verify that the correct columns have been selected.

# select the columns to use
var_selectedcolumns = ['Notice_No', 'Details', 'Act']

# Create a new dataframe with those columns
var_selected_df = var_df_gazettes[var_selectedcolumns].copy()

# Display the first 10 rows
var_selected_df.head(10)

### **2.Data Pre-processing (Cleaning)**
#### **2.1 Handling Missing Values**
We first check for missing values across the selected features.  

- **Act**:Rows with missing values are dropped to preserve data integrity.  
- **Details**: As the primary input feature, rows missing this information are also dropped, since they would not contribute meaningful input to the model.  

# Check missing values before cleaning
print('Missing values before cleaning')
print(var_selected_df.isnull().sum())