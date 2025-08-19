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



##  Extract Columns Gazette number, Act, Act number and Main body

def parse_gazette_text(txt_path):
    """
    Reads the extracted text file and parses it to separate gazette notices.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        print("Parsing extracted text...")

        # A pattern to match a full gazette notice block
        notice_pattern = r"(Gazette Notice No\. \d+ oF \d+ .*?)(?=Gazette Notice No\. \d+ oF \d+|\Z)"
        notices = re.findall(notice_pattern, full_text, re.DOTALL)

        gazette_data = []

        for notice in notices:
            # 1. Extract the Gazette Number
            gazette_number_match = re.search(r"Gazette Notice No\. (\d+)(?: oF | of )(\d+)", notice, re.IGNORECASE)
            gazette_number = f"No. {gazette_number_match.group(1)} of {gazette_number_match.group(2)}" if gazette_number_match else 'N/A'

            # 2. Extract the Act Number and Title
            act_match = re.search(r"The (.+) Act\s+\(?(No\..+of.+)\)?", notice, re.DOTALL)
            act_title = act_match.group(1).strip() if act_match else 'N/A'
            act_number = act_match.group(2).strip() if act_match else 'N/A'

            # 3. Get the Main Body
            main_body = notice.replace(gazette_number_match.group(0), '', 1) if gazette_number_match else notice
            main_body = main_body.replace(act_match.group(0), '', 1) if act_match else main_body
            main_body = main_body.strip()

            gazette_data.append({
                "Gazette Number": gazette_number,
                "Act": f"The {act_title} Act" if act_title != 'N/A' else 'N/A',
                "Act Number": act_number,
                "Main Body": main_body
            })

        return gazette_data
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None
