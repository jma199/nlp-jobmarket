import os
import codecs
from bs4 import BeautifulSoup
import csv

from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Text, insert, select, delete

def get_raw_data(directory):
    '''Open file containing html of job description and prepare soup object.'''
    fileList = []
    soupList = []
    # Iterate through each file in directory
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            # add each filename to list
            fileList.append(file)
            #print(fileList)
            # open and load html
            with codecs.open(directory + "/"+ file, 'r', "utf-8") as f:
                job_html = f.read()
                job_soup = BeautifulSoup(job_html, "html.parser")
                soupList.append(job_soup)
    print("soup_list is done.")
    return soupList

# From the loaded text, extract job information using beautiful soup
def get_job_record(job_soup):
    '''Create a record of information for one job.'''
    # Title
    try:
        job_title = job_soup.find('h1').text.strip()
    except:
        job_title = "NaN"
    
    # Company
    try:
        company = job_soup.find("div", class_="jobsearch-InlineCompanyRating").next_element.next_element.text.strip()
    except:    
        try:
            company = job_soup.find("div", class_="jobsearch-InlineCompanyRating").text.strip()
        except:
            company = "NaN"

    # Location
    try:
        job_location = job_soup.find("div", class_ = "jobsearch-InlineCompanyRating").next_sibling.text.strip()
    except:
        try:
            job_location = job_soup.find("div", class_ = "jobsearch-InlineCompanyRating").next_sibling.next_sibling.text.strip()
        except:
            job_location = 'NaN'

    # Job Description
    try:
        job_description = job_soup.find("div", class_="jobsearch-jobDescriptionText").text.strip()
    except:
        job_description = "NaN"

    # Not all postings have a salary available
    try:
        job_salary = job_soup.find("span", class_="icl-u-xs-mr--xs").text.strip()
    except AttributeError:
        job_salary = "NaN"
    
    job_record = {'jobtitle': job_title,
                                 'company': company,
                                 'location': job_location,
                                 'salary': job_salary,
                                 'jobdescription': job_description,
                                 'label': 1
                 }

    return job_record

def main_etl(directory):
    '''This function loads text data, extracts pertinent job information, and saves data in a csv file.'''
    #while True:
    soupList = get_raw_data(directory)
        
    # add each job record to a list
    # this will create a list of dictionaries, making it easy to insert into a sql table
    job_records = []
    for job_soup in soupList:
        job_record = get_job_record(job_soup)
        job_records.append(job_record)
        print("Added to job_records list. Length of job_records is: ", len(job_records))

    # add job records to sqlite db
    # Create engine: engine
    engine = create_engine('sqlite:///joblist.sqlite')
    metadata = MetaData()

    # Define a new table
    data = Table('data', metadata,
                 Column('jobtitle', String(100)),
                 Column('company', String(100)),
                 Column('location', String(25)),
                 Column('salary', Integer()),
                 Column('jobdescription', Text()),
                 Column('label', Integer())
                )

    # Create table
    metadata.create_all(engine)

    # Print table details
    print(engine.table_names())

    # Build an insert statement to insert a record into the data table: insert_stmt
    insert_stmt = insert(data)

    # Execute the insert statement via the connection: results
    connection = engine.connect()
    results = connection.execute(insert_stmt, job_records)

    # Print result rowcount
    print("The number of rows added is: ", results.rowcount)