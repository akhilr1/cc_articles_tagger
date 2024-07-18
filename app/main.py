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
from langchain.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
import functools
import logging
import time, tiktoken, faiss ,json
from langchain_huggingface import HuggingFaceEmbeddings

enc = tiktoken.get_encoding("o200k_base")
embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")

load_dotenv()
es_host=os.getenv("ES_HOST")


es = Elasticsearch(hosts=es_host)
gpt_model = os.getenv("GPT_MODEL")
base_url = os.getenv("BASE_URL")
api_type = "azure"
api_version = os.getenv("API_VERSION")
api_key = os.getenv("API_KEY")


llm = AzureChatOpenAI(
azure_endpoint=base_url,
openai_api_version=api_version,
azure_deployment=gpt_model,
openai_api_key=api_key,
openai_api_type=api_type,
model=gpt_model)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Entering {func.__name__} function")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Exiting {func.__name__} function. Time taken: {elapsed_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"Exception in {func.__name__} after {elapsed_time:.4f} seconds: {e}")
            raise
    return wrapper

@log_decorator
def retrievesummary(newdb):
    retriever=newdb.as_retriever()
    docs = retriever.invoke("get me all data")


    prompt = ChatPromptTemplate.from_template(
     """
        I need to retrieve the published date from the context. for example the passage might have something like  Last Updated : "Nov 21 2023 | 10:58 PM IST", so you should take everything after Updated as the published date.       
        Also Summarize the following text article, ensuring all specified company names and published dates mentioned in the article are explicitly included and no essential information is omitted:

        Passage:
        {input}
        """
            )
    prompt2=ChatPromptTemplate.from_template(
        """
         
           Please provide a summary of the article using the following format and give output in json format:
           1.summary: Summarize the following text article, ensuring all specified company names and published dates mentioned in the article are explicitly included and no essential information is omitted.
           2.sentiment: Describe the overall sentiment of the article. Example: 'Positive', 'Negative', 'Neutral'.
           3.sentiment_score: Assign a sentiment score to the article, ranging from -10 (strongly negative) to 10 (strongly positive). Example: -7 for strongly negative sentiment, 5 for moderately positive sentiment.
           4.sentiment_justification: Provide a one-line sentence justifying the sentiment of the article. Example: 'The article highlights significant environmental damage caused by the company.'
           5.company_name: Identify the company the article is primarily about. Example: 'Tesla'.
           6.company_name_justification: Provide a one-line sentence justifying the reason for selecting the company name. Example: 'The article discusses Tesla's new sustainability initiatives in detail.'
           7.esg_relevance: Rate how pertinent the text is to ESG matters, on a scale from 1 to 10. Example: 8 for highly relevant content.
           8.published_date: State the published date of the article in yyyy-mm-dd format. Example: '2023-06-28'.
           9.title: Provide a title for the text. Example: 'Tesla's Latest Efforts in Sustainability'.
           10.financial_data_check: Rate how pertinent the text is to stock-related data, on a scale from 1 to 10. Example: 3 for minimally relevant content.
           11.company_name_registered: State the exact registered name of the company, expanded to full form. Example: 'Tesla, Inc.'
           12.company_aliases: All possible aliases and variations of the company name the article is majorly about. Example: 'TSLA, Tesla Motors'

           Article:
           {input}
           """
          )
    prompt3=ChatPromptTemplate.from_template(
        """
          Please provide a summary of the article using the following format:
          summary: Summarize the following text article, ensuring all specified company names and published dates mentioned in the article are explicitly included and no essential information is omitted.
          Sentiment: Describe the overall sentiment of the article. Example: 'Positive', 'Negative', 'Neutral'.
          Sentiment Score: Assign a sentiment score to the article, ranging from -10 (strongly negative) to 10 (strongly positive). Example: -7 for strongly negative sentiment, 5 for moderately positive sentiment.
          Sentiment Justification: Provide a one-line sentence justifying the sentiment of the article. Example: 'The article highlights significant environmental damage caused by the company.'
          Company Name: Identify the company the article is primarily about. Example: 'Tesla'.
          Company Name Justification: Provide a one-line sentence justifying the reason for selecting the company name. Example: 'The article discusses Tesla's new sustainability initiatives in detail.'
          ESG Relevance: Rate how pertinent the text is to ESG matters, on a scale from 1 to 10. Example: 8 for highly relevant content.
          Published Date: State the published date of the article in yyyy-mm-dd format. Example: '2023-06-28'.
          Title: Provide a title for the text. Example: 'Tesla's Latest Efforts in Sustainability'.
          Financial Data Check: Rate how pertinent the text is to stock-related data, on a scale from 1 to 10. Example: 3 for minimally relevant content.
          Company Name Registered: State the exact registered name of the company, expanded to full form. Example: 'Tesla, Inc.'
          Company Aliases: List all possible aliases and variations of the company name the article is majorly about. Example: 'TSLA, Tesla Motors'
          Article:
          {input}
          """
          )


    chain = prompt2 | llm
    
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(docs[0].page_content))
    
    response=chain.invoke({"input": docs[0].page_content}).dict()
   # response['token_length']=num_tokens
    content_dict = json.loads(response['content'])
    content_dict['token_length'] = num_tokens
    response['content'] = json.dumps(content_dict)
    return response['content']





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

@log_decorator
def embedtext(text):  
 
    docs=[Document(page_content=text)]

    vectorstore = FAISS.from_documents(docs , embedding = embedding_function)
    return vectorstore

################################################################

@log_decorator
def get_es_articles(last_processed_docid):
    # es = Elasticsearch(hosts=es_host)
    index_name= "cc_dump_*"
    doc_limit=os.getenv("DOC_LIMIT")
    query = {
    "size": 10,
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

@log_decorator
def store_in_es_controverys(df):
    try:
    # Replace NaN values with appropriate defaults
        index=os.getenv("ES_INDEX")
        
        documents = df.to_dict(orient="records")
        for doc in documents:
          for key, value in doc.items():
              if pd.isna(value):
                  doc[key] = None
        actions = [
            {
                "_op_type": "index",
                "_index": index,
                "_id": doc.get("url"),
                "_source": doc
            }
            for doc in documents
        ]
        success, failed = bulk(es, actions=actions, index=index)
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
@log_decorator
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
                        vectorstore=embedtext(clean_text)
                        if vectorstore:
                            response = json.loads(retrievesummary(vectorstore))
                            print(f"length={len(articles)} index={index} doc_id={doc_id}")
                            print(f"Response: {response}")
                            articles_extracted.loc[index,'content'] = response['summary']
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
                            articles_extracted.loc[index,'token_length'] = response['token_length']
                        else:
                            logger.error(f"No vectorstore value for index={index}")




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
