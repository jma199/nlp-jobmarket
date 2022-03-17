# nlp-jobmarket Repository

An analysis of data jobs in the Toronto area

## Problem Statement

The objective of this project is to get a better understanding of the data analytics job market through exploratory data analysis of job postings in Toronto, Canada, my ideal location, and predictive modeling. Of the various job titles that exist in the data analytics field, this project will focus on data analyst and data scientist jobs.

Businesses are generating more data than ever before and are looking to generate real-time insights and predictions to optimize their performance. There are many skills and available tools that a job opening may advertise, but it is not possible to learn everything. Knowing what skills are needed can direct a job seeker to which type of job to apply for or consider further education to improve in-demand skills.


## Data Collection

Job postings were scraped from Indeed.com. From each job posting, the title, company, salary (if possible), and the job description were extracted and stored in a sqlite database.


## Exploratory Data Analysis

Before data analysis, duplicate job descriptions were removed (some were present over multiple days of scraping). 

For a given search, other job titles with overlapping skills and similar duties were present in the search results and were also removed.

The dominent sector was finance in the top 10 for both data analyst and data scientist jobs.

Using skills-based matching, the top skills in data analyst job postings included Excel, SQL/database, Tableau/PowerBI, research, and ad-hoc work. The top skills in data scientist job postings included Python/R, machine learning, SQL, and statistics.

## Predictive Modeling

A logistic regression model was used to classify job descriptions as either data analyst or data scientist. Logistic regression is a supervised machine learning model that can learn words that differentiate between our two classes. Text was pre-processed using a TF-IDF vectorizer to encode words in a job description.

The model did a good job of correctly classifying job descriptions (F1 = 0.79, AUC ROC = 0.968), though the most important words (features) of the model were not skills.

## Summary

In this project, I wanted to gain a better understanding of the Toronto data job market.

Text of the job descriptions was prepared for modeling using TF-IDF. This model correctly identified all data scientist positions (class 1) it believed to be class 1, but was unable to classify half of the actual data science postings correctly. Preparing text using the bag-of-words (BoW) technique (not shown) was also attempted and resulted in a F1-score of 0.89 and a ROC AUC of 0.964. BoW predicted class 1 vs class 0 more equally. Here, predicting Data Scientist jobs was considered extremely important, so TF-IDF results were presented.

Improvements to the logistic regression model include performing additional text preprocessing steps such as stemming to reduce noise. Including bi-grams were tried but results often included pairings of words already having their own high score of importance as individual words.

Interestingly, most important words (features) of the model were not skills. This indicates that specific hard/technical skills are not the best indicators of a job type. Using a larger dataset may improve the ability of a model to identify skills.

We could try a more advanced technique such as deep learning to develop a neural network. Depending on the size of your neural network, it could take a long time to train. 