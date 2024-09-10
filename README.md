# Azure OpenAI Agent for Self-RAG

This repo shows how to build a Self-RAG agent using Azure OpenAI, Azure AI Search and Azure App service.

Self-RAG is a strategy for RAG that incorporates self-reflection / self-grading on retrieved documents and generations.

For simplicity purposes, the application implements the agent using the OpenAI Python SDK's chat completion API and function calling. This provides extended customization possibilities as well as more control over the orchestration, prompts etc.

![image](selfrag.png)

## Pre-requisites
This repo assumes you already have the following resources deployed:
1. Azure OpenAI resource and deployment (ideally GPT-4o)
2. Azure Cognitive Search

### 1. Populate .env file with your credentials

Open the [.env.sample](./.env.sample) file and replace the placeholders with your Azure OpenAI and Search credentials, save the file an name it `.env`.

### 2. Install dependencies
Create a virtual environment with Python 3.10 or above and run the following command:
```bash
pip install -r requirements.txt
```

### 3. [Streamlit Application](./agent-selfrag-app/)

Run the Streamlit app using the following command:

```bash
streamlit run ./agent-selfrag-app/app.py
```
