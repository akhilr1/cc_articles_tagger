from calendar import month_name
from venv import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field,validator,conint
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import re
from elasticsearch import Elasticsearch
import pandas as pd
from elasticsearch.helpers import bulk, BulkIndexError
from datetime import datetime
from es_pandas import es_pandas
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()
es_host=os.getenv("ES_HOST")


es = Elasticsearch(hosts=es_host)

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    sentiment_score: conint(ge=-10, le=10) = Field(..., description="The sentiment score of the text ranging from -10 to 10")
    sentiment_justification: str = Field(description="A one-line sentence justifying the sentiment of the article")
    company_name: str = Field(description="Company name the article is majorly about")
    company_name_justification: str = Field(description="A one-line sentence justifying the reason for selecting the company name")
    esg_relevance: int = Field(description="How pertinent is the text to ESG matters, on a scale from 1 to 10?")
    published_date: str = Field(description="Published date of this article in yyyy-mm-dd")
    title: str = Field(description="Give a title for this text")
    financial_data_check: int = Field(description="How pertinent is the text to stock related data, on a scale from 1 to 10?")
    company_name_registered: str = Field(description="The exact registered name of the company, expanded to full form (e.g., Ltd to Limited)")
    company_aliases: str = Field(description="All possible aliases and variations of the company name the article is majorly about")



gpt_model = "gpt-4o"
base_url = os.getenv("BASE_URL")
api_type = "azure"
api_version = "2024-02-01"
api_key = os.getenv("API_KEY")


llm = AzureChatOpenAI(
azure_endpoint=base_url,
openai_api_version=api_version,
azure_deployment=gpt_model,
openai_api_key=api_key,
openai_api_type=api_type,
model=gpt_model).with_structured_output(
Classification
        )

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
#     Classification
# )
tagging_chain = tagging_prompt | llm


##############################################################
def clean_text_content(text):
    # Define patterns to identify and remove non-content sections
    ad_patterns = [
        r'CHANGE LANGUAGE', r'WATCH LIVE TV', r'DOWNLOAD [A-Z]+ APP', r'Follow Us On',
        r'Trending Topics', r'Home', r'Movies', r'Cricket', r'India', r'Showsha', r'Politics',
        r'World', r'Viral', r'Business', r'Education', r'Opinion', r'Photos', r'Videos',
        r'Explainers', r'Web Stories', r'Tech', r'Auto', r'Lifestyle', r'Health', r'Travel',
        r'Food', r'Sports', r'Markets', r'Tax', r'Cryptocurrency', r'Savings and Investments',
        r'Breaking News', r'AQI', r'Power Circuit', r'Elections', r'Movie Reviews', r'City News',
        r'Astrology', r'Viral', r'Bollywood', r'Hollywood', r'Regional Cinema', r'Tamil Cinema',
        r'Telugu Cinema', r'Web-series', r'Television', r'Latest', r'WTC Final', r'Health & Fitness',
        r'Showsha', r'Opinion', r'Photos', r'Videos', r'Tags:', r'first published:', r'last updated:',
        r'Latest News', r'Follow us:', r'© Copyright', r'All rights reserved', r'Sensex', r'Nifty', r'Nifty Midcap', r'Nifty Smallcap', r'Nifty Bank'
    ]

    # Combine patterns into a single regular expression
    ad_regex = re.compile('|'.join(ad_patterns), re.IGNORECASE)

    # Split the text into lines
    lines = text.split('\n')

    # Filter out lines that match advertisement patterns or have fewer than three words
    content_lines = [line for line in lines if not ad_regex.search(line) and len(line.split()) >= 4]

    # Join the cleaned lines back into a single string
    cleaned_text = '\n'.join(content_lines).strip()

    return cleaned_text
###############################################################

def get_es_articles(last_processed_docid):
    # es = Elasticsearch(hosts=es_host)
    index_name= "cc_dump_*"
    doc_limit=os.getenv("DOC_LIMIT")
    query = {
    "size": 100,
    "query": {
        "bool": {
            "filter": []
        }
    },
    "sort": [
        {"doc_id": {"order": "asc"}}
        ]
    }

    if last_processed_docid > 0:
        query["query"]["bool"]["filter"].append({"range": {"doc_id": {"gt": last_processed_docid, "lte": doc_limit}}})            
    results = es.search(index=index_name, body=query)
    articles = []
    for doc in results["hits"]["hits"]:
        articles.append(doc["_source"])

    articles_df = pd.DataFrame(articles)
    return articles_df
#####################################################################

def store_in_es_controverys(df):
    try:
    # Replace NaN values with appropriate defaults
        
        documents = df.to_dict(orient="records")
        for doc in documents:
          for key, value in doc.items():
              if pd.isna(value):
                  doc[key] = None
        actions = [
            {
                "_op_type": "index",
                "_index": "cc_classified_articles_1",
                "_id": doc.get("url"),
                "_source": doc
            }
            for doc in documents
        ]
        success, failed = bulk(es, actions=actions, index="cc_classified_articles_1")
        logger.info(f"Successfully indexed {success} documents.")
        if failed:
            logger.error(f"{failed} document(s) failed to index.")
            # Iterate through actions to find failed documents
            for action in actions:
                try:
                    es.index(index=action["_index"], id=action["_id"], document=action["_source"])
                except Exception as e:
                    logger.error(f"Failed to index document {action['_id']}: {e}")
        return success == len(documents)
    except BulkIndexError as bulk_error:
        logger.error(f"Bulk index error: {bulk_error}")
        for error in bulk_error.errors:
            logger.error(f"Failed document: {error}")
        return False
    except Exception as e:
        logger.error(f"Exception in store_in_es_controverys: {e}")
        return False


def get_docid(file_path):
    try:
        with open(file_path, "r") as file:
            doc_id = int(file.read().strip())
            return doc_id
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def set_docid(file_path, doc_id):
    try:
        with open(file_path, "w") as file:
            file.write(str(doc_id))
    except Exception as e:
        print(f"Error writing to the file: {e}")



    
######################################################################
def main():
    file_path = "/app/processed_doc_id.txt"

    #get doc_id from mongodb collection named collection_doc
    processed_docid=get_docid(file_path)
    while  True:
        articles=get_es_articles(processed_docid)
        if articles.empty == False:
            logger.info(processed_docid)
            articles_extracted= articles.drop(columns=['warc_length','content']).copy()
            for i, (index, row) in enumerate(articles.iterrows()):
                cc_content=row["content"]
                doc_id=row["doc_id"]
                clean_text=clean_text_content(cc_content)                        


                if len(clean_text)<15000:
                    try:

                        response=tagging_chain.invoke({"input": clean_text}).dict()
                        print(f"length={len(articles)} index={index}")
                        # print(clean_text)
                        print(response)
                        articles_extracted.loc[index,'content'] = clean_text
                        articles_extracted.loc[index,'sentiment'] = response['sentiment']
                        articles_extracted.loc[index,'company_names'] = response['company_name']
                        articles_extracted.loc[index,'esg_relevance'] = response['esg_relevance']
                        if response['published_date']:
                            articles_extracted.loc[index, 'published_date'] = datetime.strptime(response['published_date'], "%Y-%m-%d").date()
                        articles_extracted.loc[index,'title'] = response['title']
                        # articles_extracted.loc[index,'controvesry_category'] = response['controvesry_category']
                        articles_extracted.loc[index,'financial_data_check'] = response['financial_data_check']
                        articles_extracted.loc[index,'company_name_registered'] = response['company_name_registered']
                        articles_extracted.loc[index,'company_aliases'] = response['company_aliases']



                    except Exception as e:
                        print(f"error :{e}")
                        continue
            # articles.to_csv('articles_processed.csv', index=False)
            result=store_in_es_controverys(df=articles_extracted)
            processed_docid=doc_id
            set_docid(file_path,processed_docid)



            # break
        else:
            print(processed_docid)
            if processed_docid>0:
                logger.info("All CC documents processed")
            else:
                logger.info("No CC articles found")
            break
    

  

if __name__ == "__main__":
    main()

