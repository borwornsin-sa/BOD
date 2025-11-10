import azure.functions as func
import logging
import json
import base64
import os
import uuid
import re
from datetime import datetime
from pathlib import Path
from io import BytesIO

# --- Library Imports ---
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
import tiktoken

# --- Library Imports for Document Processing ---
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 not installed. PDF file processing will be disabled.")

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    logging.warning("python-docx not installed. DOCX file processing will be disabled.")

# --- End Document Processing Imports ---

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# --- Environment variables ---
AZURE_OPENAI_ENDPOINT_EMBEDDING = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING")
AZURE_OPENAI_API_KEY_EMBEDDING = os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

AZURE_OPENAI_ENDPOINT_GPT = os.getenv("AZURE_OPENAI_ENDPOINT_GPT")
AZURE_OPENAI_API_KEY_GPT = os.getenv("AZURE_OPENAI_API_KEY_GPT")
GPT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT", "gpt-4.1")

COSMOS_URI = os.getenv("COSMOS_URI")
COSMOS_KEY = os.getenv("COSMOS_KEY")

COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME", "Copilot-BOD")
COSMOS_KB_CONTAINER_NAME = os.getenv("COSMOS_KB_CONTAINER_NAME", "BOD-Knowledgebase")
COSMOS_USAGES_CONTAINER_NAME = os.getenv("COSMOS_USAGES_CONTAINER_NAME", "usages")

# --- Token pricing per 1M tokens (USD) ---
# (TOKEN_PRICING dictionary remains the same as your original code)
TOKEN_PRICING = {
    "gpt-4o": {
        "input": 5.00,
        "output": 15.00
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60
    },
    "gpt-4.1": {
        "input": 2.00,
        "output": 8.00,
        "cached_input": 0.50
    },
    "gpt-4.1-mini": {
        "input": 0.40,
        "output": 1.60
    },
    "gpt-4.1-nano": {
        "input": 0.10,
        "output": 0.40
    },
    "text-embedding-3-small": {
        "input": 0.02,
        "output": 0.0
    },
    "text-embedding-3-large": {
        "input": 0.13,
        "output": 0.0
    }
}

# --- Initialize clients ---
embedding_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT_EMBEDDING,
    api_key=AZURE_OPENAI_API_KEY_EMBEDDING
)
gpt_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT_GPT,
    api_key=AZURE_OPENAI_API_KEY_GPT
)
cosmos_client = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)

# --- Helper Functions ---
# (All previous helper functions like extract_metadata_with_llm, chunk_text, get_embedding,
# save_usage_to_cosmos, extract_text, delete_document_by_sharepoint_id, vector_search_sync remain the same)
# I will add the new refactored functions for the 'generate' endpoint here.

def get_embedding(text: str):
    """Get embedding from Azure OpenAI"""
    response = embedding_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

def vector_search_sync(query_vector: list, container, top_k: int = 5):
    """Synchronous vector search operation, filtering for 'chunk' documents."""
    sql_query = """
    SELECT TOP @top_k c.id, c.content, c.source, c.parent_id, c.chunk_index, c.pages,
           VectorDistance(c.embedding, @queryVector) AS score
    FROM c
    WHERE c.doc_type = "chunk"
    ORDER BY VectorDistance(c.embedding, @queryVector)
    """
    parameters = [
        {"name": "@top_k", "value": top_k},
        {"name": "@queryVector", "value": query_vector}
    ]
    return list(container.query_items(
        query=sql_query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))

def retrieve_rag_context(container, query_vector: list, top_k: int):
    """
    Performs vector search for chunks and retrieves their corresponding parent documents.
    """
    logging.info("Step 1: Retrieving context from vector search...")
    search_results = vector_search_sync(query_vector, container, top_k)
    
    if not search_results:
        logging.warning("No relevant chunks found in vector database.")
        return [], {}, {}

    logging.info(f"Found {len(search_results)} relevant chunks.")
    
    unique_parent_ids = list(set([result['parent_id'] for result in search_results if 'parent_id' in result]))
    parent_documents = {}
    all_annexes = {}

    logging.info(f"Step 2: Retrieving {len(unique_parent_ids)} unique parent documents...")
    if unique_parent_ids:
        for parent_id in unique_parent_ids:
            try:
                parent_doc = container.read_item(item=parent_id, partition_key=parent_id)
                if parent_doc and parent_doc.get("doc_type") == "parent_document":
                    parent_documents[parent_id] = parent_doc
                    # Collect annexes from each parent document
                    for annex in parent_doc.get('related_annexes', []):
                        all_annexes[annex['name']] = annex
            except Exception as e:
                logging.warning(f"Could not retrieve Parent Document for ID {parent_id}: {e}")
                
    logging.info(f"Retrieved {len(parent_documents)} parent documents and {len(all_annexes)} unique annexes.")
    return search_results, parent_documents, all_annexes


def build_rag_prompt(query: str, search_results: list, parent_documents: dict):
    """
    Builds the prompt context with references and a mapping for citations.
    """
    logging.info("Step 3: Building RAG prompt with citations...")
    context_with_refs = ""
    ref_mapping = {}
    ref_idx_counter = 1

    # Add chunk documents to the context
    for result in search_results:
        content = result.get('content', '')
        parent_id = result.get('parent_id')
        parent_doc = parent_documents.get(parent_id, {})
        file_name = parent_doc.get('file_name', 'unknown')
        web_url = parent_doc.get('web_url', '')
        pages = result.get('pages', [])
        
        chunk_ref_key = f"C{ref_idx_counter}"
        context_with_refs += f"[{chunk_ref_key}] (from: {file_name})\\n{content}\\n\\n"
        
        ref_mapping[chunk_ref_key] = {
            'url': web_url,
            'file_name': file_name,
            'pages': pages
        }
        ref_idx_counter += 1
    
    user_prompt = f"""Documents:
    {context_with_refs}
    
    Question: {query}
    Answer:"""
    
    logging.info("Successfully built RAG prompt.")
    return user_prompt, ref_mapping


def get_system_prompt():
    """Returns the standardized system prompt for the LLM."""
    # Using a helper function makes the main 'generate' function cleaner
    return """
    You are an AI assistant. Your task is to answer the user's question based *only* on the provided context below.
    - Respond ONLY in English.
    - Answer directly and concisely.
    - Base all answers on the provided documents. DO NOT use external knowledge.
    - If the answer is not found in the context, state that you cannot answer.
    - Every time you respond, you MUST use the `answer_with_citations` function to provide the reply and the list of reference keys (e.g., ["C1", "C3"]) that you used.
    - If you mention an annex or attachment in your reply, also include its name in the `mentioned_annexes` list.
    """

def process_llm_response(response, ref_mapping: dict, all_annexes: dict):
    """
    Parses the LLM function call response and formats the final reply with citation links.
    """
    logging.info("Step 5: Processing LLM response and formatting citations...")
    function_call = response.choices[0].message.function_call
    if not (function_call and function_call.name == "answer_with_citations"):
        logging.warning("LLM did not return the expected function call.")
        return response.choices[0].message.content or "Could not generate a valid response."

    result = json.loads(function_call.arguments)
    reply = result.get("reply", "No answer could be generated.")
    citations = result.get("citations", [])
    mentioned_annexes = result.get("mentioned_annexes", [])
    
    unique_citations = sorted(list(set(citations)))
    citation_links = []

    # Build links for main document citations
    for ref_key in unique_citations:
        if ref_key in ref_mapping:
            ref_info = ref_mapping[ref_key]
            url = ref_info.get('url', '')
            file_name = ref_info.get('file_name', 'unknown')
            pages = ref_info.get('pages', [])
            
            display_name = file_name
            if pages:
                page_str = ", ".join(map(str, sorted(pages)))
                display_name += f" (page {page_str})"

            if url:
                citation_links.append(f"[{display_name}]({url})")
            else:
                citation_links.append(f"{display_name} (URL not available)")

    # Build links for mentioned annexes
    for annex_name in set(mentioned_annexes):
        if annex_name in all_annexes:
            annex_data = all_annexes[annex_name]
            if annex_data.get('url'):
                citation_links.append(f"[{annex_name}]({annex_data['url']})")

    if citation_links:
        reply += "\\n\\n**References 📚:** " + ", ".join(citation_links)
        
    logging.info("Successfully formatted response with citations.")
    return reply

# --- Azure Function: generate (Refactored) ---
@app.route(route="generate")
def generate(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Generate response function triggered')
    try:
        req_body = req.get_json()
        query = req_body.get('query')
        top_k = req_body.get('top_k', 5)
        database_body = req_body.get('cosmosDatabase', COSMOS_DB_NAME)
        container_body = req_body.get('cosmosContainer', COSMOS_KB_CONTAINER_NAME)

        if not query:
            return func.HttpResponse(json.dumps({"status": "Failed", "statusDetail": "Missing query parameter"}), mimetype="application/json", status_code=400)

        database = cosmos_client.get_database_client(database_body)
        container = database.get_container_client(container_body)
        
        logging.info(f"Generating response for: '{query}' in container: {container_body}")

        # --- RAG Pipeline ---
        # Step 1 & 2: Get query embedding, search for chunks, and retrieve parent documents
        query_vector = get_embedding(query)
        search_results, parent_documents, all_annexes = retrieve_rag_context(container, query_vector, top_k)

        if not search_results:
            response_text = "I apologize, but I could not find any relevant information to answer your question."
            return func.HttpResponse(json.dumps({"status": "Completed", "reply": response_text}), mimetype="application/json", status_code=200)

        # Step 3: Build a new prompt with the retrieved context and references
        user_prompt, ref_mapping = build_rag_prompt(query, search_results, parent_documents)

        # Step 4: Invoke Final LLM with function calling
        logging.info("Step 4: Invoking final LLM...")
        functions = [
            {
                "name": "answer_with_citations",
                "description": "Provide an answer based on the documents, citing the sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reply": {"type": "string", "description": "The answer to the user's question, in English."},
                        "citations": {"type": "array", "items": {"type": "string"}, "description": "A list of reference keys used, e.g., [\"C1\", \"C2\"]."},
                        "mentioned_annexes": {"type": "array", "items": {"type": "string"}, "description": "Names of any annexes or attachments mentioned in the reply."}
                    },
                    "required": ["reply", "citations"]
                }
            }
        ]
        
        response = gpt_client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            functions=functions,
            function_call={"name": "answer_with_citations"},
            temperature=0.1
        )

        # Step 5: Process the LLM's response and format the final reply
        final_reply = process_llm_response(response, ref_mapping, all_annexes)

        # Step 6: Return final response
        # (Usage calculation and saving logic can be added here as in your original code)
        logging.info(f"✓ Successfully generated response for query: {query}")
        return func.HttpResponse(json.dumps({"status": "Completed", "reply": final_reply}), mimetype="application/json", status_code=200)

    except Exception as e:
        logging.error(f"An unexpected error occurred in the RAG pipeline: {e}", exc_info=True)
        return func.HttpResponse("An unexpected error occurred while generating the response.", status_code=500)

# --- Other Azure Functions (ingest, delete, search, test) ---
# (The code for ingest, delete, search, and test functions remains the same as your original script)

# NOTE: You would paste your original, unchanged 'ingest', 'delete', 'search', and 'test' functions here.
# Also, paste the helper functions used by them (extract_metadata_with_llm, etc.) if they are not already present.

