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
```
#### **2.3 Text Pre-processing**

To prepare the **`Details`** column for modeling, we apply a text preprocessing pipeline consisting of the following steps:  

1. **Convert to lowercase** – standardizes text and reduces duplication of words with different cases.  
2. **Remove punctuation** – eliminates punctuation marks that do not contribute to text meaning.  
3. **Remove extra whitespace** – cleans up irregular spacing caused by punctuation removal or formatting.  
4. **Remove stopwords** – removes common words (e.g., *the*, *is*) that do not carry meaningful information.  
5. **Lemmatize text** – reduces words to their base form to unify similar terms (e.g., *running* → *run*).  
6. **Full pipeline** – applies all the above steps sequentially for consistent and clean textual data.  

The cleaned text is stored in a new column **`CleanedDetails`**, allowing us to retain the original text while preparing it for modeling.  
We also verify the first few rows and the shape of the processed dataframe to confirm that the pipeline works correctly.  

---

```python
# Convert to lowercase
def fxn_convert_to_lowercase(var_text):
    return var_text.lower()

# Remove punctuation
def fxn_remove_punctuation(var_text):
    return "".join([ch if ch not in string.punctuation else " " for ch in var_text])

# Remove extra whitespace
def fxn_remove_extra_whitespace(var_text):
    return " ".join(var_text.split())

# Remove stopwords
def fxn_remove_stopwords(var_text):
    var_tokens = word_tokenize(var_text)
    var_stop_words = set(stopwords.words("english"))
    var_filtered_tokens = [word for word in var_tokens if word.lower() not in var_stop_words]
    return " ".join(var_filtered_tokens)

# Lemmatize text
def fxn_lemmatize_text(var_text):
    var_tokens = word_tokenize(var_text)
    lemmatizer = WordNetLemmatizer()
    var_lemmatized_tokens = [lemmatizer.lemmatize(word) for word in var_tokens]
    return " ".join(var_lemmatized_tokens)

# Full preprocessing pipeline
def fxn_preprocess_text_pipeline(var_text):
    if not isinstance(var_text, str):
        return ""
    text = fxn_convert_to_lowercase(var_text)
    text = fxn_remove_punctuation(text)
    text = fxn_remove_extra_whitespace(text)
    text = fxn_remove_stopwords(text)
    text = fxn_lemmatize_text(text)
    text = fxn_remove_extra_whitespace(text)  # final cleanup
    return text

# Apply pipeline to dataframe
var_processed_df["CleanedDetails"] = var_processed_df["Details"].apply(fxn_preprocess_text_pipeline)

print("--- Text Pre-processing Complete ---")
print(var_processed_df[["Details", "CleanedDetails"]].head())
var_processed_df.shape
```
## **Data Transformation**

### **Step 1: One-Hot Encoding of the `Act` Column**

Because the **`Act`** column contains nominal data, we apply one-hot encoding:  

1. Convert all text in **`Act`** to lowercase to ensure consistency.  
2. Use `OneHotEncoder` to transform each unique category in **`Act`** into a binary column.  
3. Create a new dataframe with the encoded columns.  
4. Concatenate the encoded columns with the original dataframe, aligning rows correctly.  

This transformation allows the categorical **`Act`** feature to be used effectively in machine learning models.  
We then verify the first few rows and the shape of the transformed dataframe to confirm the encoding is complete.  

---

```python
# Initialize one-hot encoder
var_encoder = OneHotEncoder()

# Convert 'Act' text to lowercase for consistency
var_processed_df["Act"] = var_processed_df["Act"].apply(fxn_convert_to_lowercase)

# Apply one-hot encoding to 'Act'
var_encoded_data = var_encoder.fit_transform(var_processed_df[['Act']])

# Create a dataframe from the encoded data
var_encoded_df = pd.DataFrame(
    var_encoded_data.toarray(), 
    columns=var_encoder.get_feature_names_out(['Act'])
)

# Align the index to match the original dataframe
var_encoded_df.index = var_processed_df.index

# Concatenate encoded columns with the original dataframe
var_transformed_df = pd.concat([var_processed_df, var_encoded_df], axis=1)

# Drop the original 'Act' and 'Details' columns
var_transformed_df = var_transformed_df.drop(columns=['Act', 'Details'])

# Display the first 5 rows transposed for easier readability
var_transformed_df.head(5).T
```

### **Transforming Text**
#### **Bag-of-Words (BoW)**
To utilise the  **CleanedDetails** column in our model, we use the Bag of Words with these parameters :

- **Max features = 300**: We limit the number of features to 300 to reduce dimensionality and avoid sparsity, while keeping the most informative terms.  
- **N-gram range = (1,2)**: We include both unigrams and bigrams to capture important word combinations that convey more meaning than single words alone.  
- **Minimum document frequency = 2**: We remove very rare words that appear in fewer than 2 documents, as they are unlikely to contribute to general patterns.  

After fitting and transforming the text, we create a dataframe from the BoW matrix and concatenate it with the original dataframe.  
Finally, we drop the original **CleanedDetails** column because its numeric representation is now included, keeping the dataset ready for modeling.  
We display the first 20 rows (transposed) to verify the transformation.

``` python
# Initialize CountVectorizer with max 300 features, unigrams and bigrams, min 2 docs
var_bow_vectorizer = CountVectorizer(
    max_features=300,
    ngram_range=(1,2),
    min_df=2,
)

# Fit and transform 'CleanedDetails' to BoW matrix
var_matrix = var_bow_vectorizer.fit_transform(var_transformed_df['CleanedDetails'])

# Create dataframe from BoW matrix
var_bow_df = pd.DataFrame(var_matrix.toarray(), columns=var_bow_vectorizer.get_feature_names_out())

# Concatenate BoW features with original dataframe
var_final_processed_df = pd.concat([var_transformed_df, var_bow_df], axis=1)

# Drop the original 'CleanedDetails' column
var_final_processed_df = var_final_processed_df.drop(columns=['CleanedDetails'])

# Display first 20 rows transposed
var_final_processed_df.head(20).T
```
## **MODELING**
## **dataset splitting with k fold**
More Reliable Performance Estimate:
A single train-test split can be misleading. The model's performance might be high or low purely by chance, depending on which data points ended up in the training and testing sets. K-fold cross-validation mitigates this by running the evaluation process multiple times.
Efficient Use of Data:

---


It allows you to use your entire dataset for both training and validation. Every data point gets to be in a test set exactly once and in a training set k-1 times. This is especially beneficial for smaller datasets where a single train-test split might leave you with too little data for either training or testing.

Reduces Overfitting:

---


The cross-validation process helps detect if your model is overfitting. If the model performs well on the training folds but poorly on \the validation folds, it's a strong sign of overfitting.

``` python

# Drop rows with missing 'Notice_Type' before defining features and target
var_final_processed_df = var_final_processed_df.dropna(subset=['Notice_Type'])

var_final_processed_df.head()

# Define the target variable
var_target = var_final_processed_df['Notice_Type']

# Drop the target variable and 'Notice_No' from the features
var_features = var_final_processed_df.drop(columns=['Notice_Type', 'Notice_No'])

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over the folds and train the model
for fold, (train_index, test_index) in enumerate(kf.split(var_features, var_target)):


    # Split data into training and testing sets for this fold
    X_train, X_test = var_features.iloc[train_index], var_features.iloc[test_index]
    y_train, y_test = var_target.iloc[train_index], var_target.iloc[test_index]

```
## **model 1 training - random forest**
Key Benefits Of Random Forest

    High Accuracy: By averaging out the biases of individual trees, a Random Forest often produces highly accurate predictions.

    Reduced Overfitting: The random sampling of data and features for each tree ensures that the model doesn't get too specific to the training set.

    Feature Importance: You can easily determine which features were most influential in the model's predictions, providing valuable insights into your data.

    Handles Complex Data: It can model complex, non-linear relationships in your data without requiring extensive feature engineering.
``` python
# Initialize and train a RandomForestClassifier model for this fold
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Drop the 'CleanedDetails' column from the full feature set before the k-fold split
var_features_numeric = var_features.drop(columns=['CleanedDetails'], errors='ignore')

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over the folds and train the model
for fold, (train_index, test_index) in enumerate(kf.split(var_features_numeric, var_target)):

    # Split data into training and testing sets for this fold using the numeric features
    X_train, X_test = var_features_numeric.iloc[train_index], var_features_numeric.iloc[test_index]
    y_train, y_test = var_target.iloc[train_index], var_target.iloc[test_index]


    forest_model.fit(X_train, y_train)

    print(f"Fold {fold+1}:")
    print(f"  Train data shape: {X_train.shape}")
    print(f"  Test data shape: {X_test.shape}")
    print("-" * 20)

print("Model training complete for all folds.")
```
## **model 2 training**
### Data Preparation and Feature Engineering for Classification

``` python



# Prepare dataset
var_processed_df = var_processed_df.dropna(subset=["CleanedDetails", "Notice_Type", "Act"])

X_text = var_processed_df["CleanedDetails"]
X_act = var_processed_df[["Act"]]
y = var_processed_df["Notice_Type"]


# Train/Test Split
X_text_train, X_text_test, X_act_train, X_act_test, y_train, y_test = train_test_split(
    X_text, X_act, y, test_size=0.3, random_state=42
)


# initialize the tfidfvectorizer
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = vectorizer.transform(X_text_test)

# Act Features (One-Hot)
var_Hotencoder = OneHotEncoder(handle_unknown="ignore")
var_X_act_train = var_Hotencoder.fit_transform(X_act_train)
var_X_act_test = var_Hotencoder.transform(X_act_test)

# Combine Features
X_train_combined = hstack([X_text_train_tfidf, var_X_act_train])
X_test_combined = hstack([X_text_test_tfidf, var_X_act_test])


# Train Naive Bayes
var_Bayes = MultinomialNB()
var_Bayes.fit(X_train_combined, y_train)


```


###**Model 1 Predictions**
Model Performance Evaluation Using Cross-Validation

Here we assess the performance of the machine learning model using cross-validation, a robust technique for estimating a model's generalization ability.

The process involves these key steps:

   Defining Variables: The code first defines the target variable (var_target) to be predicted and the features (var_features) used for prediction. It explicitly drops columns not needed for the model, such as the target and a unique identifier (Notice_No).

   Running Cross-Validation: It then uses cross_val_score to split the dataset into five folds (cv=5). The model is trained on four folds and tested on the remaining fold, repeating this process five times. This ensures the model is evaluated on different subsets of the data, providing a more reliable performance estimate than a single train-test split.

  Displaying Results: Finally, the code prints the accuracy score for each of the five folds and calculates the mean cross-validation accuracy. This average provides a single, more stable measure of the model's expected performance on unseen data.
```

    ``` python



from sklearn.model_selection import cross_val_score

# Define the target variable
var_target = var_final_processed_df['Notice_Type']

# 'Notice_No' from the features
var_features = var_final_processed_df.drop(columns=['Notice_Type', 'Notice_No'])


var_score = cross_val_score(forest_model, var_features, var_target, scoring='accuracy', cv=5)
print(f"Cross-validation accuracy scores: {var_score}")
print(f"Mean cross-validation accuracy: {var_score.mean():.2f}")

print("Confusion Matrix of the First model")
# Assuming 'model', 'var_features', and 'var_target' are already defined
y_pred = cross_val_predict(forest_model, var_features, var_target, cv=5)

# Compute confusion matrix
cm = confusion_matrix(var_target, y_pred, labels=var_target.unique())

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=var_target.unique())
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=90)
plt.title("Confusion Matrix (Cross-Validated Predictions)")
plt.show()

```

###**Model 2 Predictions**

Evaluating a Naive Bayes Classifier for text classification using several performance metrics.

Firstly we predict the labels (y_pred) for the test data (X_test_tfidf) using a trained Naive Bayes model (nb). It then prints a comprehensive summary of the model's performance, including:

  Accuracy Score: A simple metric showing the proportion of correctly classified instances.

  Classification Report: Provides a detailed breakdown of precision, recall, and F1-score for each class.

  Confusion Matrix: A visual representation  that helps understand where the model is making errors, by showing the counts of true positive, true negative, false positive, and false negative predictions. The matrix is plotted as a heatmap for easy interpretation.

''' python
