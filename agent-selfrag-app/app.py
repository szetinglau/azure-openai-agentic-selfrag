from dotenv import load_dotenv
import os, requests
from openai import AzureOpenAI
import pandas as pd
import streamlit as st
import urllib
import json
import io
import numpy as np
import time
from azure.search.documents.models import VectorizedQuery, VectorizableTextQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType

load_dotenv()

azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
deployment = os.getenv('AZURE_OPENAI_MODEL_DEPLOYMENT')
search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"])
search_service_name = os.environ["AZURE_SEARCH_SERVICE"]
index_name = os.environ["AZURE_SEARCH_INDEX"]


REWRITE_PROMPT = """You are a helpful assistant. You help users search for the answers to their questions.
Rewrite the following question into a more useful search query to find the most relevant documents.
Example:
===
scalable storage solution
steps to create a scalable storage solution
===
"""

client = AzureOpenAI(
    api_version="2023-03-15-preview",
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
)
search_client = SearchClient(endpoint=search_endpoint, index_name='cleveland-index', credential=credential)

# If you are not using a supported model or region, you may not be able to use json_object response format
# Please see https://learn.microsoft.com/azure/ai-services/openai/how-to/json-mode
def rewrite_query(query: str):
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Input: {query}"}
        ],
        max_tokens=100
    )
    try:
        #print(response)  # Debug print to inspect the response structure
        response_content = response.choices[0].message.content
        #print(response_content)  # Debug print to check content before JSON parsing
        return response_content
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        raise

def search_azure_search(query: str):
    try:
        results = search_client.search( query_type='semantic',
            search_text=query,
            select="content",
            semantic_configuration_name='azureml-default',
            query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
            top=2
        )
        content = ''
        for result in results:
             content += result['content'] + '\n\n'
        return content
        
    except Exception as e:
        print("An error occurred:", e)


# Define the functions you provided
def grade_documents(docs: str, search_query: str) -> str:
    """
    Grade the relevance of documents using Azure OpenAI.

    Args:
        docs (str): The retrieved documents.
        query (str): The user query.

    Returns:
        str: 'generate' if documents are relevant, 'rewrite' if they are not.
    """
    sys_prompt = """
    You are a grader assessing relevance of a retrieved document to a user question.
    
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    usr_prompt = f""" Here is the retrieved document: {docs}"""
    
    # Combine the document and query into a single prompt
    # full_prompt = f"{prompt}\n\nUser Question: {search_query}"
    messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt}
        ]

    response = client.chat.completions.create(
        model=deployment,  # Use the correct model name for your deployment
        messages=messages,  # Use prompt instead of messages
        max_tokens=50  # Limit the token output
    )
    
    score = response.choices[0].message.content.strip().lower()

    if score == "yes":
        return "generate"
    else:
        return "rewrite_query"

def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_azure_search",
                "description": "Search azure search index for relevant documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The name of the table to get the schema for",
                        },
                    },
                    "required": ["query"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grade_documents",
                "description": "Grade the relevance of documents using Azure OpenAI",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docs": {
                            "type": "string",
                            "description": "The document content to be assessed for relevance.",
                        },
                        "search_query": {
                            "type": "string",
                            "description": "The original search query that the document should be assessed against.",
                        },
                    },
                    "required": ["docs", "search_query"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rewrite_query",
                "description": "Rewrite the following question into a useful search query to find the most relevant documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Rewrite the following question into a useful search query to find the most relevant documents",
                        },
                    },
                    "required": ["query"],
                },
            }
        }
    ]

def get_available_functions():
    return {
        "search_azure_search":search_azure_search,
        "grade_documents":grade_documents,
        "rewrite_query":rewrite_query
        }

@st.cache_data
def init_system_prompt():
    return [
    {"role":"system", "content":"""
     You are an intelligent agent capable of answering user questions by retrieving relevant information from both external knowledge sources (like databases and document stores) and generating human-like responses based on your retrieval.

     First, perform the azure search directly on the given query.
     Then, grade the query results to determine if the documents are relevant to the user query USING GRADE DOCUMENTS FUNCTION.
     If yes, generate a response. If not, rewrite the query to perform a more effective search.
     Keep REWRITING the query until you find relevant documents.
 
    Think step by step, before doing anything, share the different steps you'll execute to get the answer
    Think step by step, before doing anything, share the different steps you'll execute to get the answer
     """}
    ]

def reset_conversation():
  st.session_state.messages = []

st.set_page_config(page_title="Azure Agentic Agent")
st.title("Chat with Agentic")

st.info("This is a simple chat app to demo how to create a database agent powered by Azure OpenAI and capable of interacting with Azure SQL", icon="📃")


st.button('Clear Chat History 🔄', on_click=reset_conversation)


# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = deployment

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


system_prompt = init_system_prompt()

def get_message_history():
    return system_prompt+st.session_state.messages
            

function_responses = {}
def process_stream(stream):
    # Empty container to display the assistant's reply
    assistant_reply_box = st.empty()
    
    # A blank string to store the assistant's reply
    assistant_reply = ""
    # To hold function responses
    

    # Iterate through the stream
    tool_calls = []
    for event in stream:
        # Here, we only consider if there's a delta text
        delta = event.choices[0].delta if event.choices and event.choices[0].delta is not None else None
        if delta and delta.content:
            # empty the container
            assistant_reply_box.empty()
            # add the new text
            assistant_reply += delta.content
                # display the new text
            assistant_reply_box.markdown(assistant_reply)
            

        elif delta and delta.tool_calls:
            tc_chunk_list = delta.tool_calls
            for tc_chunk in tc_chunk_list:
                if len(tool_calls) <= tc_chunk.index:
                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

                tc = tool_calls[tc_chunk.index]
                if tc_chunk.id:
                    tc["id"] += tc_chunk.id
                if tc_chunk.function.name:
                    tc["function"]["name"] += tc_chunk.function.name
                if tc_chunk.function.arguments:
                    tc["function"]["arguments"] += tc_chunk.function.arguments    
    
    if assistant_reply!="":
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    if tool_calls:
        st.session_state.messages.append({"role": "assistant", "tool_calls": tool_calls})
        available_functions = get_available_functions()

        for tool_call in tool_calls:
            # Note: the JSON response may not always be valid; be sure to handle errors
            function_name = tool_call['function']['name']

            print("\nCalling function name\n", function_name)

            # Step 3: call the function with arguments if any
            function_to_call = available_functions[function_name]
            # print("\nFunction to call\n", function_to_call)
            function_args = json.loads(tool_call['function']['arguments'])
            # print("\nFunction args\n", function_args)
            with st.status(f"Running function: {function_name}...", expanded=True) as status:
                
                if function_name == "search_azure_search":
                    # Run search_azure_search and store its result
                    #print("\nFunction args\n", function_args)
                    function_response = function_to_call(**function_args)
                    #print("\nFunction response\n", function_response)
                    function_responses['search'] = function_response
                    #print("\nFunction responses\n", function_responses)
                    status.write(f"Search function outputs: {function_response}")
                elif function_name == "grade_documents":
                    # Ensure `search_azure_search` was called before grading documents
                    search_results = function_responses.get('search', {'search': []})
                    # print("\nfunction_responses\n", function_responses)
                    if search_results:
                        # Pass documents to grade_documents
                        function_args['docs'] = search_results
                        print("\nFunction args\n", function_args)
                        function_response = function_to_call(**function_args)
                        print("\nFunction response\n", function_response)
                        status.write(f"Grade function outputs: {function_response}")
                elif function_name == "rewrite_query":
                    # print("Rewriting query...")
                    #print("\nFunction args\n", function_args)
                    function_response = function_to_call(**function_args)
                    #print("\nFunction response\n", function_response)
                    status.write(f"Query outputs: {function_response}")
                status.update(label=f"Function {function_name} completed!", state="complete", expanded=False)


            st.session_state.messages.append(
                {
                    "tool_call_id": tool_call['id'],
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
           
        return True
    else:
        
        return False

    
    

messages = st.container(height=400, border=False)
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"] and 'content' in message:
        with messages.chat_message(message["role"]):
            st.markdown(message["content"])
# Display chat input
prompt=st.chat_input("Ask me anything...")


with messages:
# Accept user input
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with messages.chat_message("user"):
            st.markdown(prompt)

    # Display assistant response in chat message container
        with st.chat_message("assistant"):
            has_more = True
            while has_more:
                
                stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=get_message_history(),
                stream=True,
                tools = get_tools(),
                tool_choice="auto"
            )
                has_more = process_stream(stream)
