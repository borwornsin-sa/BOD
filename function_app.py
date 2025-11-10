import azure.functions as func
import logging
import json
import base64
import os
import uuid
from datetime import datetime
from pathlib import Path
from io import BytesIO
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
# Default Cosmos DB Name and new Knowledge Base Container Name
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME", "Copilot-BOD")
COSMOS_KB_CONTAINER_NAME = os.getenv("COSMOS_KB_CONTAINER_NAME", "BOD-Knowledgebase")
COSMOS_USAGES_CONTAINER_NAME = os.getenv("COSMOS_USAGES_CONTAINER_NAME", "usages") # Your existing usages container
# --- Token pricing per 1M tokens (USD) ---
# (TOKEN_PRICING dictionary remains the same as your original code)
TOKEN_PRICING = {
    "gpt-4o": {
        "input": 2.00,
        "output": 8.00
    },
    "gpt-4o-mini": {
        "input": 0.40,
        "output": 1.60
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

def get_embedding(text: str):
    """Get embedding from Azure OpenAI"""
    response = embedding_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=text)
    return response.data[0].embedding

def vector_search_sync(query_vector: list, container, top_k: int = 10):
    """Synchronous vector search operation, filtering for 'chunk' documents."""
    sql_query = "SELECT TOP @top_k c.id, c.content, c.source, c.parent_id, c.chunk_index, c.pages, VectorDistance(c.embedding, @queryVector) AS score FROM c WHERE c.doc_type = 'chunk' ORDER BY VectorDistance(c.embedding, @queryVector)"
    parameters = [{"name": "@top_k", "value": top_k}, {"name": "@queryVector", "value": query_vector}]
    return list(container.query_items(query=sql_query, parameters=parameters, enable_cross_partition_query=True))

def retrieve_rag_context(container, query_vector: list, top_k: int):
    """Performs vector search for chunks and retrieves their corresponding parent documents and annexes."""
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
                    for annex in parent_doc.get('related_annexes', []):
                        all_annexes[annex['name']] = annex
            except Exception as e:
                logging.warning(f"Could not retrieve Parent Document for ID {parent_id}: {e}")
                
    logging.info(f"Retrieved {len(parent_documents)} parent documents and {len(all_annexes)} unique annexes.")
    return search_results, parent_documents, all_annexes

def build_rag_prompt(query: str, search_results: list, parent_documents: dict, all_annexes: dict):
    """Builds the prompt with context, citations, and annex awareness."""
    logging.info("Step 3: Building RAG prompt with citations and annex awareness...")
    context_with_refs = ""
    ref_mapping = {}
    ref_idx_counter = 1

    for result in search_results:
        content, parent_id = result.get('content', ''), result.get('parent_id')
        parent_doc = parent_documents.get(parent_id, {})
        file_name, web_url, pages = parent_doc.get('file_name', 'unknown'), parent_doc.get('web_url', ''), result.get('pages', [])
        chunk_ref_key = f"C{ref_idx_counter}"
        context_with_refs += f"[{chunk_ref_key}] (from: {file_name})\n{content}\n\n"
        ref_mapping[chunk_ref_key] = {'url': web_url, 'file_name': file_name, 'pages': pages}
        ref_idx_counter += 1
    
    annex_awareness_prompt = ""
    if all_annexes:
        annex_names = ", ".join(f"'{name}'" for name in all_annexes.keys())
        annex_awareness_prompt = f"The following annexes/attachments are related to these documents: {annex_names}. If the user's question is about one of these, or if the context mentions them, be sure to include their names in your reply.\n\n"

    user_prompt = f"Documents:\n{context_with_refs}\n---\n{annex_awareness_prompt}Question: {query}"
    
    logging.info("Successfully built RAG prompt.")
    return user_prompt, ref_mapping

def get_system_prompt():
    """Returns the standardized system prompt for the LLM."""
    return """You are an AI assistant. Your task is to answer the user's question based *only* on the provided context below.
- Respond ONLY in Thai.
- Answer directly and concisely.
- Base all answers on the provided documents. DO NOT use external knowledge.
- If the answer is not found in the context, state that you cannot answer.
- Every time you respond, you MUST use the `answer_with_citations` function to provide the reply and the list of reference keys (e.g., ["C1", "C3"]) that you used.
- If you mention an annex or attachment in your reply, also include its name in the `mentioned_annexes` list."""

def process_llm_response(response, ref_mapping: dict, all_annexes: dict):
    """Parses the LLM function call and formats the final reply with citation links."""
    logging.info("Step 5: Processing LLM response and formatting citations...")
    function_call = response.choices[0].message.tool_calls[0].function
    if not (function_call and function_call.name == "answer_with_citations"):
        logging.warning("LLM did not return the expected function call.")
        return response.choices[0].message.content or "Could not generate a valid response."

    result = json.loads(function_call.arguments)
    reply = result.get("reply", "No answer could be generated.")
    citations = result.get("citations", [])
    mentioned_annexes = result.get("mentioned_annexes", [])
    
    unique_citations = sorted(list(set(citations)))
    citation_links = []

    for ref_key in unique_citations:
        if ref_key in ref_mapping:
            ref_info = ref_mapping[ref_key]
            url, file_name, pages = ref_info.get('url', ''), ref_info.get('file_name', 'unknown'), ref_info.get('pages', [])
            display_name = file_name
            if pages:
                page_str = ", ".join(map(str, sorted(pages)))
                display_name += f" (page {page_str})"
            if url:
                citation_links.append(f"[{display_name}]({url})")
            else:
                citation_links.append(f"{display_name} (URL not available)")

    for annex_name in set(mentioned_annexes):
        if annex_name in all_annexes:
            annex_data = all_annexes[annex_name]
            if annex_data.get('url'):
                citation_links.append(f"[{annex_name}]({annex_data['url']})")
    
    if citation_links:
        reply += "\n\n**References 📚:** " + ", ".join(citation_links)
        
    logging.info("Successfully formatted response with citations.")
    return reply

# --- Start of Functions that were missing ---

def extract_text(file_content: bytes, file_name: str) -> str:
    """Extracts text from PDF or DOCX files."""
    file_extension = Path(file_name).suffix.lower()
    text = ""
    if file_extension == '.pdf' and PDF_SUPPORT:
        with BytesIO(file_content) as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    elif file_extension == '.docx' and DOCX_SUPPORT:
        with BytesIO(file_content) as f:
            doc = docx.Document(f)
            for para in doc.paragraphs:
                text += para.text + '\n'
    else:
        # Fallback for other text-based files or if libraries are missing
        try:
            text = file_content.decode('utf-8')
        except UnicodeDecodeError:
            logging.warning(f"Could not decode file {file_name} as UTF-8.")
            text = ""
    return text

def chunk_text(text: str, model: str, max_tokens: int = 8000):
    """Splits text into chunks based on token count."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# This is a placeholder; you might have a more complex implementation
def extract_metadata_with_llm(text: str):
    """Extracts structured metadata from text using an LLM call (placeholder)."""
    logging.info("Extracting metadata with LLM (placeholder implementation)...")
    # In a real scenario, you would make a call to gpt_client here
    # For this example, we'll return a dummy structure.
    return {
        "title": "Extracted Title Placeholder",
        "related_annexes": [{"name": "Annex 1 Placeholder.pdf", "url": ""}]
    }

def delete_document_by_sharepoint_id(sharepoint_id: str, container):
    """Deletes a parent document and all its chunks from Cosmos DB."""
    logging.info(f"Attempting to delete document with SharePoint ID: {sharepoint_id}...")
    # Find the parent document first
    query = "SELECT * FROM c WHERE c.sharepoint_id = @sp_id AND c.doc_type = 'parent_document'"
    params = [{"name": "@sp_id", "value": sharepoint_id}]
    parent_docs = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))

    if not parent_docs:
        logging.warning(f"No parent document found with SharePoint ID: {sharepoint_id}")
        return 0
    
    parent_doc = parent_docs[0]
    parent_id = parent_doc['id']
    
    # Find all chunks associated with the parent
    chunk_query = "SELECT c.id FROM c WHERE c.parent_id = @parent_id AND c.doc_type = 'chunk'"
    chunk_params = [{"name": "@parent_id", "value": parent_id}]
    chunks_to_delete = list(container.query_items(chunk_query, parameters=chunk_params, enable_cross_partition_query=True))
    
    deleted_count = 0
    # Delete chunks
    for chunk in chunks_to_delete:
        container.delete_item(item=chunk['id'], partition_key=chunk['id'])
        deleted_count += 1
    
    # Delete parent document
    container.delete_item(item=parent_id, partition_key=parent_id)
    deleted_count += 1
    
    logging.info(f"Successfully deleted {deleted_count} items for SharePoint ID: {sharepoint_id}")
    return deleted_count

# --- Azure Function: generate (The one we modified) ---

@app.route(route="generate")
def generateBODPRD(req: func.HttpRequest) -> func.HttpResponse:
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
        
        query_vector = get_embedding(query)
        search_results, parent_documents, all_annexes = retrieve_rag_context(container, query_vector, top_k)

        if not search_results:
            response_text = "I apologize, but I could not find any relevant information to answer your question."
            return func.HttpResponse(json.dumps({"status": "Completed", "reply": response_text}), mimetype="application/json", status_code=200)

        user_prompt, ref_mapping = build_rag_prompt(query, search_results, parent_documents, all_annexes)

        logging.info("Step 4: Invoking final LLM...")
        tools = [{"type": "function", "function": {"name": "answer_with_citations", "description": "Provide an answer based on the documents, citing the sources.", "parameters": {"type": "object", "properties": {"reply": {"type": "string", "description": "The answer to the user's question, in Thai."}, "citations": {"type": "array", "items": {"type": "string"}, "description": "A list of reference keys used, e.g., [\"C1\", \"C2\"]."}, "mentioned_annexes": {"type": "array", "items": {"type": "string"}, "description": "Names of any annexes or attachments mentioned in the reply."}}, "required": ["reply", "citations"]}}}]
        
        response = gpt_client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=[{"role": "system", "content": get_system_prompt()}, {"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "answer_with_citations"}},
            temperature=0.1
        )
        
        final_reply = process_llm_response(response, ref_mapping, all_annexes)
        
        logging.info(f"✓ Successfully generated response for query: {query}")
        return func.HttpResponse(json.dumps({"status": "Completed", "reply": final_reply}), mimetype="application/json", status_code=200)

    except Exception as e:
        logging.error(f"An unexpected error occurred in the RAG pipeline: {e}", exc_info=True)
        return func.HttpResponse("An unexpected error occurred while generating the response.", status_code=500)

# --- Other Azure Functions (ingest, delete, search, test) ---

@app.route(route="ingest")
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Ingest function triggered')
    try:
        req_body = req.get_json()
        file_name = req_body['file_name']
        file_content_b64 = req_body['file_content_b64']
        sharepoint_id = req_body['sharepoint_id']
        web_url = req_body.get('web_url', '')

        file_content = base64.b64decode(file_content_b64)
        
        database = cosmos_client.get_database_client(COSMOS_DB_NAME)
        container = database.get_container_client(COSMOS_KB_CONTAINER_NAME)

        # First, delete any existing document with the same SharePoint ID
        delete_document_by_sharepoint_id(sharepoint_id, container)

        # 1. Extract text from the document
        text = extract_text(file_content, file_name)
        if not text:
            return func.HttpResponse(json.dumps({"status": "Failed", "statusDetail": "Could not extract text from file."}), mimetype="application/json", status_code=400)

        # 2. Extract metadata using LLM (using the placeholder function)
        metadata = extract_metadata_with_llm(text[:10000]) # Use a sample for speed

        # 3. Create parent document
        parent_id = str(uuid.uuid4())
        parent_doc = {
            'id': parent_id,
            'sharepoint_id': sharepoint_id,
            'doc_type': 'parent_document',
            'file_name': file_name,
            'web_url': web_url,
            'title': metadata.get('title'),
            'related_annexes': metadata.get('related_annexes', []),
            'ingested_at': datetime.utcnow().isoformat()
        }
        container.upsert_item(parent_doc)

        # 4. Chunk text and create chunk documents
        text_chunks = chunk_text(text, GPT_DEPLOYMENT)
        chunk_docs_to_upload = []
        for i, chunk in enumerate(text_chunks):
            chunk_id = str(uuid.uuid4())
            chunk_doc = {
                'id': chunk_id,
                'parent_id': parent_id,
                'doc_type': 'chunk',
                'chunk_index': i,
                'content': chunk,
                'embedding': get_embedding(chunk),
                # You might add page numbers here if `extract_text` provides them
                'pages': [] 
            }
            container.upsert_item(chunk_doc)
        
        logging.info(f"Successfully ingested {file_name} as {len(text_chunks)} chunks.")
        return func.HttpResponse(json.dumps({"status": "Completed", "parent_id": parent_id, "chunks_created": len(text_chunks)}), mimetype="application/json", status_code=200)

    except Exception as e:
        logging.error(f"Error during ingestion: {e}", exc_info=True)
        return func.HttpResponse(f"An error occurred during ingestion: {str(e)}", status_code=500)


@app.route(route="delete")
def delete(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Delete function triggered')
    try:
        req_body = req.get_json()
        sharepoint_id = req_body['sharepoint_id']
        
        database = cosmos_client.get_database_client(COSMOS_DB_NAME)
        container = database.get_container_client(COSMOS_KB_CONTAINER_NAME)
        
        deleted_count = delete_document_by_sharepoint_id(sharepoint_id, container)
        
        if deleted_count > 0:
            return func.HttpResponse(json.dumps({"status": "Completed", "items_deleted": deleted_count}), mimetype="application/json", status_code=200)
        else:
            return func.HttpResponse(json.dumps({"status": "Not Found", "detail": f"No document with SharePoint ID {sharepoint_id} found."}), mimetype="application/json", status_code=404)

    except Exception as e:
        logging.error(f"Error during deletion: {e}", exc_info=True)
        return func.HttpResponse(f"An error occurred during deletion: {str(e)}", status_code=500)


@app.route(route="search")
def search(req: func.HttpRequest) -> func.HttpResponse:
    """A simple vector search endpoint, returns raw chunk data."""
    logging.info('Search function triggered')
    try:
        req_body = req.get_json()
        query = req_body.get('query')
        top_k = req_body.get('top_k', 5)

        if not query:
            return func.HttpResponse("Missing 'query' parameter.", status_code=400)

        database = cosmos_client.get_database_client(COSMOS_DB_NAME)
        container = database.get_container_client(COSMOS_KB_CONTAINER_NAME)
        
        query_vector = get_embedding(query)
        results = vector_search_sync(query_vector, container, top_k)
        
        return func.HttpResponse(json.dumps(results, indent=2), mimetype="application/json", status_code=200)

    except Exception as e:
        logging.error(f"Error during search: {e}", exc_info=True)
        return func.HttpResponse(f"An error occurred during search: {str(e)}", status_code=500)

@app.route(route="test")
def test(req: func.HttpRequest) -> func.HttpResponse:
    """A simple test endpoint to check if the function app is running."""
    logging.info('Test function triggered')
    return func.HttpResponse("Hello from the RAG Function App! I am running.", status_code=200)
