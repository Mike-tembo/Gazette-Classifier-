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
