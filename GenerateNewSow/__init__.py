import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import openai
from itertools import islice
import numpy as np
import os
import time
import uuid
import pandas as pd
import re
import requests
import sys
from num2words import num2words
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
import json

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType

# Replace these with your own values, either in environment variables or directly here
AZURE_STORAGE_CONNECTIONSTRING = os.environ.get("AZURE_STORAGE_CONNECTION") or "DefaultEndpointsProtocol=https;AccountName=sowcataloguestorage;AccountKey=8A9jND/GAcL73dSGodJaQ4AiCIKZ+IQSNyggDxlp3LB0uWdscUabg0Bp6+MonXdvEOkbtTvq+z2f+AStKRsJOw==;EndpointSuffix=core.windows.net"
AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT") or "sowcataloguestorage"
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER") or "sow-generated"
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE") or "sow-search-service"
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX") or "azureblob-index"
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or "openai-sow2" # 1st "https://openai-sow.openai.azure.com/"
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "text-davinci-003"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "gpt-35-turbo"

OPENAI_API_KEY = os.environ["OpenAI_API_Key"]
#OPENAI_API_KEY = "93a83e3140f64d3285afbae12f50ac9d" #1st "e1901bf96dd14e6eaad93bef54a8bb94"

KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT") or "content"
KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY") or "category"
KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE") or "sourcepage"

# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed, 
# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the 
# keys for each service
#azure_credential = DefaultAzureCredential()
AZURE_CREDENTIAL_SEARCH = AzureKeyCredential("IkfQqdPvevny2N2HURKH0jVQi68SsZMq8H1We1xte0AzSeC2kYQf")
AZURE_CREDENTIAL_STORAGE = AzureKeyCredential("YI0tUV/AYZr8hFVxB6syqcBTNiO6DfoDlMLiiYCrMGr3YaXe/aHhGrSHRC6Ad9hkxzKkhQ3IY27K+AStCHEG7Q==")

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

COMPLETION_MODEL = 'gpt-35-turbo' 
#COMPLETION_MODEL='text-davinci-003' 

openai.api_key = OPENAI_API_KEY
openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
#openai.api_version = "2022-12-01"
openai.api_version = "2023-03-15-preview"


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # get name querystring
    queryBaseOfferId = req.params.get('BaseOfferId')
    queryCustomerName = req.params.get('CustomerName')
    queryMsAffiliate = req.params.get('MsAffiliate')
    queryContractor = req.params.get('Contractor')
    queryDesc = req.params.get('Desc')

    if not queryDesc:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            queryBaseOfferId = req_body.get('BaseOfferId')
            queryCustomerName = req_body.get('CustomerName')
            queryMsAffiliate = req_body.get('MsAffiliate')
            queryContractor = req_body.get('Contractor')
            queryDesc = req_body.get('Desc')

    query = f"{queryCustomerName}, {queryDesc}"

    if queryDesc:
        sourceDoc = searchForDoc(query)
        generatedDoc = generateDoc(query, sourceDoc)
        jsonObj = saveDoc(generatedDoc)

        return func.HttpResponse(body=jsonObj, mimetype="application/json", status_code=200)
    else:

        jsonObj = json.dumps({ "id": "00000000-0000-0000-0000-00000000", "documentName" : "", "documentUrl": "", "documentContent": ""})
        return func.HttpResponse(body=jsonObj, mimetype="application/json", status_code=200)
        
        #return func.HttpResponse(
        #     "This HTTP triggered function executed successfully. Pass keywords in the query string or in the request body for a generated document.",
        #     status_code=200
        #)

def saveDoc(docContent):
    docId = str(uuid.uuid4())
    docName = f"{docId}.html"

    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTIONSTRING)
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER)
    blob_client = blob_service_client.get_blob_client(AZURE_STORAGE_CONTAINER, docName)
    blob_client.upload_blob(docContent, blob_type="BlockBlob")

    docUrl = blob_client.url

    return json.dumps({ "id": docId, "documentName" : docName, "documentUrl": docUrl, "documentContent": ""})

def searchForDoc(searchQuery):
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX,
        credential=AZURE_CREDENTIAL_SEARCH)

    # Exclude category, to simulate scenarios where there's a set of docs you can't see
    exclude_category = None

    filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

    r = search_client.search(searchQuery, 
                            filter=filter,
                            query_type=1, 
                            #query_language="en-us", 
                            #query_speller="lexicon", 
                            #semantic_configuration_name="default", 
                            top=3)
    results = [str(doc[KB_FIELDS_SOURCEPAGE]) + ": " + doc[KB_FIELDS_CONTENT].replace("\n", "").replace("\r", "") for doc in r]
    content = "\n".join(results)

    return content

def generateDoc(query, docContent):
    # filter SOWs to most related based on embeddings, to be used in adding context to prompt

    # get text and chunk it
    chunks = [docContent[i:i+EMBEDDING_CTX_LENGTH] for i in range(0, len(docContent), EMBEDDING_CTX_LENGTH)]

    # build message and iterate on chunks to put into message
    logging.info("Sending request for outline...")

    #send first message group to start convo and get back JSON of content structure
    response = openai.ChatCompletion.create(
        messages= [
            {'role': 'system', 'content': 'You are a helpful assistant. You write documents that use other documents to pull content from. These documents are called statements of work (or SOW for short).'},
            {'role': 'user', 'content': 'Use the below Statement of Work Document to answer the subsequent question. If the answer cannot be found, write "I don\'t know." \n\n Statement of Work Document: """ '},
            {'role': 'user', 'content': str(chunks[0]) }, # we don't add context for the first 3 parts of the doc, but final would most likely summarize and then piece those together
            {'role': 'user', 'content': str(chunks[1]) },
            {'role': 'user', 'content': str(chunks[2]) },
            {'role': 'user', 'content': ' """ \n\n  Question: create an outline of this document. Output it as a semantically correct JSON array that can be used in code. Include the following attributes for each section : "title" (the heading of the section), "description" (a one sentence description of the purpose), "content" (an unsummarized output of all the text in this section)'}
        ],
        engine=COMPLETION_MODEL,
        temperature=0.2,
    )

    # print response
    jsonOutline = response['choices'][0]['message']['content']
    jsonOutline = normalize_text(jsonOutline).replace('```','')
    logging.info(jsonOutline)

    docOutline = json.loads(jsonOutline)

    logging.info("Reading outline...")
    generatedContent = []
    for section in docOutline:
            
        sectionTitle = section["title"]
        sectionDesc = section["description"]
        sectionContent = section["content"]

        logging.info("Generating content for for section '" + sectionTitle + "'...")
        
        # for each section craft a prompt to generate new content based on original parameters
        contentResponse = openai.ChatCompletion.create(
            messages= [
                {'role': 'system', 'content': 'You are a helpful software architect. You write documents that use other documents to pull content from. These documents are called statements of work (or SOW for short).'},
                {'role': 'user', 'content': f'You read a statement of work for another customer and need to use that to write content for a section that is as detailed and factual as possible because it is a contract. That section is called "{sectionTitle}" and the following statement should be used as your writing prompt to define the statement of work for that section using "{query}" as your input. Output your written content using this writing prompt: \n\n**** START PROMPT **** {sectionContent} \n\n **** END PROMPT ****" '},
                #{'role': 'user', 'content': f'Return only your written content and not any contextual information passed.'}
            ],
            engine=COMPLETION_MODEL,
            temperature=0.2,
        )

        generatedContent.append({ "title" : sectionTitle, "original-content" : sectionContent, "generated-content" : contentResponse['choices'][0]['message']['content']})

    # iterate over generated content and construct the final doc - in this case an HTML doc, but final would be a word document
    htmlContent = []
    for section in generatedContent:
        title = section['title']
        originalContent = section['original-content']
        generatedContent = section['generated-content']
        formattedContent = generatedContent.replace('\n', '<br />')

        htmlContent.append(f"<section><h2>{title}</h2><p>{formattedContent}</p></section>") #<h3>Original</h3><p>{originalContent}</p>

    htmlContentJoined = "".join(htmlContent)
    html = f"<html><head><title>AI Generated SOW Document</title></head><body>{htmlContentJoined}</body></html>"
    
    # htmlFile  = open("..\\files\\" + documentTitle + ".html", "wt")
    # n = htmlFile.write(html)
    # htmlFile.close()

    logging.info(html)
    logging.info("COMPLETE! File has been generated.")
    return html

def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.replace("#","")
    s = s.replace("|","") # custom
    s = s.replace("+-","")# custom
    s = s.replace("--","")# custom
    s = s.strip()
    
    return s
