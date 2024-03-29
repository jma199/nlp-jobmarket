# nlp-jobmarket Repository

An analysis of data jobs in the Greater Toronto area

## Problem Statement

Businesses are generating more data than ever before and are looking to generate real-time insights and predictions to optimize their performance. There are many skills and tools a job opening may advertise, and depend on a company's existing infrastructure. Knowing the most likely needed skills can influence which type of job to apply for further education to gain in-demand skills. Finding alignment between job seekers and hiring managers is critical to filling open job vacancies.

There are various job titles that exist in the data analytics field and will focus on data analyst and data scientist jobs. The objective of this project is to better understand the similarities and differences between these two job types through exploratory data analysis on job postings in Toronto, Canada and create a predictive model given job postings to predict the best job title based on a job description.


## Data Collection

Job postings were scraped from Indeed.com. From each job posting, the title, company, salary (if possible), and the job description were extracted and stored in a sqlite database.

Before data analysis, duplicate job descriptions were removed (some were present over multiple days of scraping). In addition, both 'data analyst' and 'data scientist' search results, other similar job titles with overlapping skills and duties were present and were also removed.


## Exploratory Data Analysis

There are two main points to takeaway from EDA:

1- The dominent sector was finance for both data analyst and data scientist jobs.

2- Using skills-based matching, the top skills in data analyst job postings included Excel, SQL/database, Tableau/PowerBI, research, and ad-hoc work. The top skills in data scientist job postings included Python/R, machine learning, SQL, and statistics.

## Predictive Modeling

A CatBoost Classifier model was used to classify job descriptions as either data analyst or data scientist. CatBoost is an algorithm for gradient boosting on decision trees developed by Yandex. The text was pre-processed using Count vectorizer, which removed stop words to extract text from a job description.

The model correctly classified job descriptions with an accuracy of 94%. The model precision (the proportion of positive identifications that were actually correct) was 92.9%. The recall (the proportion of actual positives that were correctly identified) was 95.1%

The most important features were machine, analyst, scientist, and python.

## Conclusion

In this project, we have explored text data from job descriptions of data analyst and data scientist job postings and built a predictive model.

Future improvements include performing additional text preprocessing steps such as stemming or lemmatization to reduce noise and including bi-grams/tri-grams.

Try it out for yourself here: 
https://jma199-st-predict-jobtitle-streamlit-app-28a177.streamlit.app/