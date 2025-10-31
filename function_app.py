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
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

AZURE_OPENAI_ENDPOINT_GPT = os.getenv("AZURE_OPENAI_ENDPOINT_GPT")
AZURE_OPENAI_API_KEY_GPT = os.getenv("AZURE_OPENAI_API_KEY_GPT")
GPT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT", "gpt-4o-mini")

COSMOS_URI = os.getenv("COSMOS_URI")
COSMOS_KEY = os.getenv("COSMOS_KEY")
# Default Cosmos DB Name and new Knowledge Base Container Name
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME", "Copilot-BOD")
COSMOS_KB_CONTAINER_NAME = os.getenv("COSMOS_KB_CONTAINER_NAME", "BOD-Knowledgebase")
COSMOS_USAGES_CONTAINER_NAME = os.getenv("COSMOS_USAGES_CONTAINER_NAME", "usages") # Your existing usages container

# Token pricing per 1M tokens (USD)
TOKEN_PRICING = {
    "gpt-4o": {
        "input": 2.00,
        "output": 8.00
    },
    "gpt-4o-mini": {
        "input": 0.40,
        "output": 1.60
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
# --- End Environment variables ---


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
# --- End Initialize clients ---


# --- Helper Functions ---
def extract_metadata_with_llm(full_text: str, file_name: str) -> dict:
    """
    Uses GPT to extract structured metadata from the full text of a document.
    Returns metadata as a dictionary, including mentioned annexes.
    """
    system_prompt = """
    You are an AI assistant designed to extract key information from meeting minutes or similar documents and return it as a JSON object.
    Based on the provided text, please extract the specified fields.
    If a field cannot be found, omit it or set it to null.
    
    The meeting_date should be in YYYY-MM-DD format.
    The participants list should include their name and role. Prioritize key roles like Chairman, Director, CEO.
    The financial_highlights should only contain numerical values. Ensure year is part of the financial highlight key, e.g., "net_profit_2024".
    The tags should be a list of relevant keywords.
    The summary should be a concise one-paragraph summary of the document.
    Additionally, identify any specific annexes or attachments mentioned in the document and list their names.

    Expected JSON structure:
    {
        "document_type": "string",
        "meeting_info": {
            "company_name": "string",
            "meeting_type": "string",
            "meeting_date": "YYYY-MM-DD"
        },
        "participants": [
            {"name": "string", "role": "string"}
        ],
        "agendas": ["string"],
        "financial_highlights": {
            "net_profit_year": "number",
            "total_assets_year": "number",
            "audit_fee_year": "number"
            // Add other relevant financial figures as needed. Use year in key.
        },
        "tags": ["string"],
        "summary": "string",
        "mentioned_annexes": ["string"] // NEW: Add this field for annexes mentioned in the text
    }
    """
    user_prompt = f"""
    Please extract the structured metadata from the following document text:
    Document Name: {file_name}
    
    Text:
    {full_text}
    
    JSON Output:
    """
    
    try:
        response = gpt_client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # Keep temperature low for precise extraction
        )
        metadata = json.loads(response.choices[0].message.content)
        return metadata
    except Exception as e:
        logging.error(f"Error extracting metadata with LLM for {file_name}: {e}")
        return {} # Return empty dict on failure


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50, page_num: int = None):
    """Split text into chunks with overlap"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({
            'text': chunk_text,
            'pages': [page_num] if page_num is not None else []
        })
        start += (chunk_size - overlap)
    return chunks

def get_embedding(text: str):
    """Get embedding from Azure OpenAI"""
    response = embedding_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

def save_usage_to_cosmos(database_name: str, operation: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float, input_text: str = None, output_text: str = None, additional_data: dict = None):
    """Save usage data to usages container"""
    try:
        database = cosmos_client.get_database_client(database_name)
        usages_container = database.get_container_client(COSMOS_USAGES_CONTAINER_NAME) # Use the specific usages container name
        usage_doc = {
            "id": str(uuid.uuid4()),
            "operation": operation,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": round(cost_usd, 8),
            "timestamp": datetime.utcnow().isoformat(),
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "time": datetime.utcnow().strftime("%H:%M:%S")
        }
        if input_text:
            usage_doc["input_text"] = input_text
        if output_text:
            usage_doc["output_text"] = output_text
        if additional_data:
            usage_doc.update(additional_data)
        
        # --- FIX: Removed 'partition_key' parameter from create_item call ---
        # The partition key value (usage_doc["date"]) is automatically picked up from the body
        # as long as the container's partition key path is set to '/date'.
        usages_container.create_item(body=usage_doc) 
        # --- END FIX ---

        logging.info(f"✓ Saved usage: {operation} - {model} - {input_tokens + output_tokens} tokens - ${cost_usd:.6f}")
    except Exception as e:
        logging.error(f"Failed to save usage data: {e}")


def extract_text(file_bytes: bytes, file_name: str):
    """Extract text from file bytes with page information"""
    file_extension = Path(file_name).suffix.lower()
    if file_extension == '.pdf':
        if not PDF_SUPPORT:
            raise ImportError("PyPDF2 not installed. Cannot process PDF files.")
        pdf_file = BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages = []
        for i, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append((i, text))
        return pages
    elif file_extension == '.docx':
        if not DOCX_SUPPORT:
            raise ImportError("python-docx not installed. Cannot process DOCX files.")
        doc_file = BytesIO(file_bytes)
        doc = docx.Document(doc_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return [(None, text)]  # DOCX doesn't have clear page boundaries
    else:
        # Plain text or other unsupported types, attempt to decode
        try:
            text = file_bytes.decode('utf-8')
            return [(None, text)]
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported file type or encoding: {file_name}")

def delete_document_by_sharepoint_id(sharepoint_item_id: str, container):
    """
    Delete all documents (Parent and Chunks) associated with a given sharepoint_item_id.
    Assumes sharepoint_item_id is the Partition Key.
    """
    try:
        # Query for all documents with the given sharepoint_item_id (this will be an In-Partition Query)
        docs_to_delete = list(container.query_items(
            query="SELECT c.id FROM c WHERE c.sharepoint_item_id = @item_id",
            parameters=[{"name": "@item_id", "value": sharepoint_item_id}],
            partition_key=sharepoint_item_id # CRUCIAL: Specify partition_key for In-Partition Query
        ))

        if not docs_to_delete:
            logging.info(f"No existing documents found for item {sharepoint_item_id} to delete.")
            return

        for doc in docs_to_delete:
            # Delete each document using its ID and the correct partition key
            container.delete_item(doc['id'], partition_key=sharepoint_item_id)
        
        logging.info(f"Deleted {len(docs_to_delete)} documents for item {sharepoint_item_id}.")
    except Exception as e:
        # Log as error if deletion fails, as it's a critical operation for updates
        logging.error(f"Failed to delete existing documents for item {sharepoint_item_id}: {e}")
        # Re-raise the exception to signal a critical failure to the caller
        raise
# --- End Helper Functions ---


# --- Azure Function: ingest ---
@app.route(route="ingest")
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Ingest function triggered')
    
    try:
        # Parse request body
        req_body = req.get_json()
        
        sharepoint_item_id = req_body.get('itemId')
        file_name = req_body.get('fileName')
        web_url = req_body.get('webUrl')
        modified_date = req_body.get('modifiedDate')
        file_content_base64 = req_body.get('fileContent')
        # NEW: Get potential_related_files from the request body (sent by Orchestrator)
        potential_related_files = req_body.get('potential_related_files', []) 

        # Get CosmosDB parameters - Ensure these point to the new knowledge base container
        custom_database = req_body.get('cosmosDatabase', COSMOS_DB_NAME) 
        custom_container = req_body.get('cosmosContainer', COSMOS_KB_CONTAINER_NAME) 
        
        database = cosmos_client.get_database_client(custom_database)
        container = database.get_container_client(custom_container)

        # Validate required fields
        if not sharepoint_item_id:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "Missing SharePoint Item ID"}),
                mimetype="application/json", status_code=400
            )

        # FIX #1: Ensure sharepoint_item_id is a string for all Cosmos DB operations
        if sharepoint_item_id is not None:
            sharepoint_item_id = str(sharepoint_item_id)

        if not file_name:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "Missing file name"}),
                mimetype="application/json", status_code=400
            )
        if not file_content_base64:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "Missing file content"}),
                mimetype="application/json", status_code=400
            )
        
        logging.info(f"Processing: {file_name} (SharePoint ID: {sharepoint_item_id}) into container {custom_container}")
        
        # Decode file content
        try:
            file_bytes = base64.b64decode(file_content_base64)
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Invalid file content encoding: {str(e)}"}),
                mimetype="application/json", status_code=400
            )
        
        # Check file size
        if len(file_bytes) == 0:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "File is empty"}),
                mimetype="application/json", status_code=400
            )
        if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"File too large: {len(file_bytes)/1024/1024:.1f}MB (max 10MB)"}),
                mimetype="application/json", status_code=400
            )
        
        # Extract text
        try:
            pages = extract_text(file_bytes, file_name)
        except ImportError as e:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Unsupported file type or missing library: {str(e)}"}),
                mimetype="application/json", status_code=400
            )
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Failed to extract text: {str(e)}"}),
                mimetype="application/json", status_code=500
            )
        
        # Check if text extraction successful
        if not pages or all(len(text.strip()) < 10 for _, text in pages):
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "No text content found in file"}),
                mimetype="application/json", status_code=400
            )
        
        # Combine all pages into full text for LLM metadata extraction
        full_text = "\n".join([page_text for page_num, page_text in pages if page_text])

        # Delete old documents (Parent and Chunks) associated with this sharepoint_item_id
        try:
            delete_document_by_sharepoint_id(sharepoint_item_id, container)
        except Exception as e:
            logging.error(f"Error deleting old documents for {sharepoint_item_id}: {str(e)}")
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Failed to delete old documents: {str(e)}"}),
                mimetype="application/json", status_code=500
            )
        
        # Chunk text per page and collect all chunks
        try:
            all_chunks = []
            for page_num, page_text in pages:
                page_chunks = chunk_text(page_text, chunk_size=500, overlap=50, page_num=page_num)
                all_chunks.extend(page_chunks)
        except Exception as e:
            logging.error(f"Error chunking text for {sharepoint_item_id}: {str(e)}") 
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Failed to chunk text: {str(e)}"}),
                mimetype="application/json", status_code=500
            )
        
        # LLM Metadata Extraction & Annex Matching Logic
        extracted_metadata = {}
        final_related_annexes = []
        if full_text:
            try:
                # Get metadata from LLM
                extracted_metadata = extract_metadata_with_llm(full_text, file_name)
                
                # Process mentioned annexes and match with potential_related_files
                mentioned_annexes = extracted_metadata.get('mentioned_annexes', [])
                
                for annex in mentioned_annexes:
                    matched_file = None
                    # Try to find matching file in potential_related_files
                    for related_file in potential_related_files:
                        if (annex.lower() in related_file.get('fileName', '').lower() or 
                            related_file.get('fileName', '').lower() in annex.lower()):
                            matched_file = related_file
                            break
                    
                    # Create annex entry with matched information or defaults
                    annex_entry = {
                        "name": annex,
                        "title": matched_file.get('fileName', annex) if matched_file else annex,
                        "url": matched_file.get('webUrl') if matched_file else None,
                        "sharepoint_item_id": matched_file.get('itemId') if matched_file else None
                    }
                    final_related_annexes.append(annex_entry)
            
            except Exception as e:
                logging.warning(f"Error during metadata extraction or annex matching: {str(e)}")
                # Continue with empty metadata rather than failing completely
                extracted_metadata = {}
                final_related_annexes = []

        # Create or Update Enriched Parent Document
        try:
            parent_doc = {
                "id": sharepoint_item_id, # Use sharepoint_item_id as ID
                "partitionKey": sharepoint_item_id, # Set Partition Key
                "doc_type": "parent_document", # Indicate type
                "file_name": file_name,
                "web_url": web_url,
                "modified_date": modified_date,
                "source": "sharepoint",
                "total_chunks": len(all_chunks), # Number of chunks generated (determined after chunking)
                "indexed_at": datetime.utcnow().isoformat(),
                "metadata": extracted_metadata, # Include LLM-extracted metadata
                "related_annexes": final_related_annexes # Use the matched annexes
            }
            container.upsert_item(body=parent_doc)
            logging.info(f"Upserted Parent Document for SharePoint ID: {sharepoint_item_id}")
        except Exception as e:
            logging.error(f"Error saving Parent Document for {sharepoint_item_id}: {str(e)}")
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Failed to save Parent Document to database: {str(e)}"}),
                mimetype="application/json", status_code=500
            )

        # Generate embeddings and save Chunk Documents
        total_embedding_tokens = 0
        try:
            for i, chunk_data in enumerate(all_chunks):
                embedding_response = embedding_client.embeddings.create(
                    model=EMBEDDING_DEPLOYMENT,
                    input=chunk_data['text']
                )
                embedding = embedding_response.data[0].embedding
                total_embedding_tokens += embedding_response.usage.total_tokens

                chunk_doc = { # Renamed 'doc' to 'chunk_doc' for clarity
                    "id": str(uuid.uuid4()),
                    "partitionKey": sharepoint_item_id, # CRUCIAL: Same as Parent Document's Partition Key
                    "doc_type": "chunk", # Indicate type
                    "parent_id": sharepoint_item_id, # Link to Parent Document
                    "sharepoint_item_id": sharepoint_item_id, # FIX #2: Add sharepoint_item_id field for Partition Key
                    "source": "sharepoint", # Keep source if useful for queries
                    "chunk_index": i,
                    "total_chunks": len(all_chunks),
                    "pages": sorted([p for p in chunk_data['pages'] if p is not None]) if chunk_data['pages'] else [], # FIX #3: Handle None in pages
                    "content": chunk_data['text'],
                    "embedding": embedding,
                    "indexed_at": datetime.utcnow().isoformat()
                }
                container.create_item(body=chunk_doc) # Create new item for each chunk
            
            embedding_pricing = TOKEN_PRICING.get(EMBEDDING_DEPLOYMENT, TOKEN_PRICING["text-embedding-3-small"])
            total_embedding_cost = (total_embedding_tokens / 1_000_000) * embedding_pricing["input"]

        except Exception as e:
            logging.error(f"Error saving Chunk Documents for {sharepoint_item_id}: {str(e)}")
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Failed to save Chunk Documents to database: {str(e)}"}),
                mimetype="application/json", status_code=500
            )
        
        # Save usage to usages container
        try:
            save_usage_to_cosmos(
                database_name=custom_database, 
                operation="ingest",
                model=EMBEDDING_DEPLOYMENT,
                input_tokens=total_embedding_tokens,
                output_tokens=0,
                cost_usd=total_embedding_cost,
                input_text=file_name,
                additional_data={
                    "sharepoint_item_id": sharepoint_item_id, # ensure it's a string here
                    "chunks_created": len(all_chunks),
                    "container": custom_container,
                    "file_size_bytes": len(file_bytes)
                }
            )
        except Exception as e:
            logging.error(f"Failed to save usage data for ingest operation: {e}")
        
        logging.info(f"✓ Successfully indexed {file_name} with SharePoint ID: {sharepoint_item_id}")
        
        return func.HttpResponse(
            json.dumps({
                "status": "Completed",
                "parentDocumentId": sharepoint_item_id,
                "chunksCreated": len(all_chunks),
                "usage": {
                    "input_tokens": total_embedding_tokens,
                    "output_tokens": 0,
                    "total_tokens": total_embedding_tokens,
                    "cost_usd": round(total_embedding_cost, 8)
                }
            }),
            mimetype="application/json", status_code=200
        )
        
    except ValueError:
        return func.HttpResponse(
            json.dumps({"status": "Failed", "statusDetail": "Invalid JSON format in request"}),
            mimetype="application/json", status_code=400
        )
    except Exception as e:
        error_message = str(e)
        logging.error(f"Unexpected error in ingest function: {error_message}")
        return func.HttpResponse(
            json.dumps({"status": "Failed", "statusDetail": f"Unexpected error: {error_message}"}),
            mimetype="application/json", status_code=500
        )
# --- End Azure Function: ingest ---


# --- Azure Function: delete ---
@app.route(route="delete")
def delete(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Delete function triggered')
    try:
        # Parse request
        req_body = req.get_json()
        sharepoint_item_id = req_body.get('itemId')
        
        # Ensure sharepoint_item_id is a string for Cosmos DB operations
        if sharepoint_item_id is not None:
            sharepoint_item_id = str(sharepoint_item_id)

        # Get optional CosmosDB parameters - Ensure this points to the knowledge base container
        custom_database = req_body.get('cosmosDatabase', COSMOS_DB_NAME)
        custom_container = req_body.get('cosmosContainer', COSMOS_KB_CONTAINER_NAME) 

        database = cosmos_client.get_database_client(custom_database)
        container = database.get_container_client(custom_container)

        # Validate required field
        if not sharepoint_item_id:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "Missing SharePoint Item ID"}),
                mimetype="application/json", status_code=400
            )
        logging.info(f"Deleting documents for SharePoint item: {sharepoint_item_id} from container {custom_container}")
        
        # Delete documents using the updated helper function
        try:
            delete_document_by_sharepoint_id(sharepoint_item_id, container)
            logging.info(f"✓ Successfully deleted documents for item {sharepoint_item_id}")
            return func.HttpResponse(
                json.dumps({"status": "Completed", "message": f"All documents for item {sharepoint_item_id} deleted."}),
                mimetype="application/json", status_code=200
            )
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Failed to delete documents: {str(e)}"}),
                mimetype="application/json", status_code=500
            )
            
    except ValueError:
        return func.HttpResponse(
            json.dumps({"status": "Failed", "statusDetail": "Invalid JSON format in request"}),
            mimetype="application/json", status_code=400
        )
    except Exception as e:
        error_message = str(e)
        logging.error(f"Unexpected error in delete function: {error_message}")
        return func.HttpResponse(
            json.dumps({"status": "Failed", "statusDetail": f"Unexpected error: {error_message}"}),
            mimetype="application/json", status_code=500
        )
# --- End Azure Function: delete ---


# --- Azure Function: search ---
def vector_search_sync(query_vector: list, container, top_k: int = 5):
    """Synchronous vector search operation, now filtering for 'chunk' documents."""
    sql_query = """
    SELECT TOP @top_k c.id, c.content, c.source, c.parent_id, c.chunk_index, c.pages, c.sharepoint_item_id,
                VectorDistance(c.embedding, @queryVector) AS score
    FROM c
    WHERE c.doc_type = "chunk" -- NEW: Filter for chunk documents only
    ORDER BY VectorDistance(c.embedding, @queryVector)
    """
    parameters = [
        {"name": "@top_k", "value": top_k},
        {"name": "@queryVector", "value": query_vector}
    ]
    # enable_cross_partition_query can be True, but ideally we'd filter by parent_id if known.
    # For initial search, cross-partition is acceptable if you don't have a specific partition key yet.
    return list(container.query_items(
        query=sql_query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))

@app.route(route="search")
def search(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Search function triggered')
    
    try:
        # Parse request
        req_body = req.get_json()
        query = req_body.get('query')
        top_k = req_body.get('top_k', 5)
        
        # CHANGE: Ensure cosmosDatabase and cosmosContainer point to the new knowledge base
        database_body = req_body.get('cosmosDatabase', COSMOS_DB_NAME)
        container_body = req_body.get('cosmosContainer', COSMOS_KB_CONTAINER_NAME)
        
        database = cosmos_client.get_database_client(database_body)
        container = database.get_container_client(container_body)

        # Validate required fields
        if not query:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "Missing query parameter"}),
                mimetype="application/json", status_code=400
            )
        if not isinstance(top_k, int) or top_k <= 0 or top_k > 50:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "top_k must be an integer between 1 and 50"}),
                mimetype="application/json", status_code=400
            )
        
        logging.info(f"Searching for: {query} (top_k: {top_k}) in container: {container_body}")
        
        # Get embedding for query
        try:
            embedding_response = embedding_client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT,
                input=query
            )
            query_vector = embedding_response.data[0].embedding
            embedding_usage = embedding_response.usage
            embedding_tokens = embedding_usage.total_tokens
            embedding_pricing = TOKEN_PRICING.get(EMBEDDING_DEPLOYMENT, TOKEN_PRICING["text-embedding-3-small"])
            embedding_cost = (embedding_tokens / 1_000_000) * embedding_pricing["input"]
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Failed to generate embedding: {str(e)}"}),
                mimetype="application/json", status_code=500
            )
        
        # Perform vector search on chunk documents
        try:
            results = vector_search_sync(query_vector, container, top_k)
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Vector search failed: {str(e)}"}),
                mimetype="application/json", status_code=500
            )
        
        # NEW: Collect unique parent_ids from search results
        unique_parent_ids = list(set([result['parent_id'] for result in results if 'parent_id' in result]))
        
        # NEW: Retrieve Parent Documents to get full metadata, file_name, web_url and related_annexes
        parent_documents = {}
        if unique_parent_ids:
            for parent_id in unique_parent_ids:
                try:
                    # Retrieve Parent Document using its ID and Partition Key
                    parent_doc = container.read_item(item=parent_id, partition_key=parent_id)
                    if parent_doc and parent_doc.get("doc_type") == "parent_document":
                        parent_documents[parent_id] = parent_doc
                except Exception as e:
                    logging.warning(f"Could not retrieve Parent Document for ID {parent_id}: {e}")

        # Format response and gather ALL annexes from retrieved Parent Documents
        formatted_results = []
        all_annexes_for_response = {} # Collect all unique annexes from parent documents

        for result in results:
            parent_id = result.get('parent_id')
            if parent_id in parent_documents:
                parent_doc_for_chunk = parent_documents[parent_id]
                # Populate file_name and web_url from the Parent Document
                file_name = parent_doc_for_chunk.get("file_name", "unknown")
                web_url = parent_doc_for_chunk.get("web_url", "")

                # Add annexes from the parent document to the overall annexes list
                for annex in parent_doc_for_chunk.get('related_annexes', []):
                    # Use annex name as key for uniqueness, storing the full annex object
                    all_annexes_for_response[annex['name']] = annex 
            else:
                file_name = "unknown"
                web_url = ""

            formatted_results.append({
                "content": result.get("content", ""),
                "source": result.get("source", ""),
                "file_name": file_name, # Now from Parent Document
                "chunk_index": result.get("chunk_index", 0),
                "web_url": web_url, # Now from Parent Document
                "pages": result.get("pages", []),
                "parent_id": result.get("parent_id"),
                "score": result.get("score", 0.0)
            })

        # Save usage to usages container
        try:
            save_usage_to_cosmos(
                database_name=database_body,
                operation="search",
                model=EMBEDDING_DEPLOYMENT,
                input_tokens=embedding_tokens,
                output_tokens=0,
                cost_usd=embedding_cost,
                input_text=query,
                additional_data={
                    "top_k": top_k,
                    "results_count": len(results),
                    "container": container_body
                }
            )
        except Exception as e:
            logging.error(f"Failed to save usage data for search operation: {e}")
        
        logging.info(f"✓ Search completed: found {len(results)} chunk results and {len(all_annexes_for_response)} unique annexes from parents.")

        # Reformat annexes for response to be a list of {name, url}
        response_annexes = [{"name": name, "url": data.get('url', '')} for name, data in all_annexes_for_response.items()]

        return func.HttpResponse(
            json.dumps({
                "status": "Completed",
                "query": query,
                "results": formatted_results,
                "annexes": response_annexes, # Return collected annexes
                "total_results": len(results),
                "usage": {
                    "input_tokens": embedding_tokens,
                    "output_tokens": 0,
                    "total_tokens": embedding_tokens,
                    "cost_usd": round(embedding_cost, 8)
                }
            }),
            mimetype="application/json", status_code=200
        )
        
    except ValueError:
        return func.HttpResponse(
            json.dumps({"status": "Failed", "statusDetail": "Invalid JSON format in request"}),
            mimetype="application/json", status_code=400
        )
    except Exception as e:
        error_message = str(e)
        logging.error(f"Unexpected error in search function: {error_message}")
        return func.HttpResponse(
            json.dumps({"status": "Failed", "statusDetail": f"Unexpected error: {error_message}"}),
            mimetype="application/json", status_code=500
        )
# --- End Azure Function: search ---


# --- Azure Function: generate ---
@app.route(route="generate")
def generate(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Generate response function triggered')
    try:
        # Parse request
        req_body = req.get_json()
        query = req_body.get('query')
        top_k = req_body.get('top_k', 5)
        
        # CHANGE: Ensure cosmosDatabase and cosmosContainer point to the new knowledge base
        database_body = req_body.get('cosmosDatabase', COSMOS_DB_NAME)
        container_body = req_body.get('cosmosContainer', COSMOS_KB_CONTAINER_NAME)
        
        # Validate required fields
        if not query:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "Missing query parameter"}),
                mimetype="application/json", status_code=400
            )
        if not database_body or not container_body:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": "Missing cosmosDatabase or cosmosContainer"}),
                mimetype="application/json", status_code=400
            )

        database = cosmos_client.get_database_client(database_body)
        container = database.get_container_client(container_body)
        
        logging.info(f"Generating response for: {query} in container: {container_body}")
        
        # Step 1: Get embedding for query and perform vector search
        try:
            query_vector = get_embedding(query)
            # This calls the updated vector_search_sync which only returns chunks
            search_results = vector_search_sync(query_vector, container, top_k)
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"Search failed: {str(e)}"}),
                mimetype="application/json", status_code=500
            )

        if not search_results:
            # Save usage for failed generation
            try:
                save_usage_to_cosmos(
                    database_name=database_body,
                    operation="generate",
                    model=GPT_DEPLOYMENT,
                    input_tokens=0, # No input tokens if no search results
                    output_tokens=0,
                    cost_usd=0.0,
                    input_text=query,
                    output_text="No relevant documents found.", # Changed to English
                    additional_data={"top_k": top_k, "container": container_body, "status_detail": "No relevant documents found"}
                )
            except Exception as e:
                logging.error(f"Failed to save usage data for generate (no search results): {e}")

            return func.HttpResponse(
                json.dumps({"status": "Completed", "reply": "No relevant documents found."}), # Changed to English
                mimetype="application/json", status_code=200
            )
        
        # NEW: Step 2.1 - Retrieve Parent Documents based on search_results
        unique_parent_ids = list(set([result['parent_id'] for result in search_results if 'parent_id' in result]))
        parent_documents = {}
        if unique_parent_ids:
            for parent_id in unique_parent_ids:
                try:
                    # Retrieve Parent Document using its ID and Partition Key
                    parent_doc = container.read_item(item=parent_id, partition_key=parent_id)
                    if parent_doc and parent_doc.get("doc_type") == "parent_document":
                        parent_documents[parent_id] = parent_doc
                except Exception as e:
                    logging.warning(f"Could not retrieve Parent Document for ID {parent_id}: {e}")

        # ------------------------------------------------------------------
        # |                     START OF CODE CHANGES                      |
        # ------------------------------------------------------------------

        # Step 2.2: Prepare context, ref_mapping, and collect all annexes
        context_with_refs = ""
        ref_mapping = {}
        all_annexes_for_response = {} # Collect all unique annexes from parent documents

        # First, add Parent Documents as references and part of context
        # For performance, we'll only include basic info in context_with_refs for parent docs
        # and rely on the LLM to understand parent_id for chunk references.
        ref_idx_counter = 1 # Start counter for references
        for p_id, p_doc in parent_documents.items():
            file_name_for_ref = p_doc.get('file_name', 'Unknown File')
            web_url_for_ref = p_doc.get('web_url', '')

            # Use a unique reference key for parent documents, e.g., 'P1', 'P2'
            parent_ref_key = f"P{ref_idx_counter}"
            # Keep parent document reference short in context to save tokens and time
            context_with_refs += f"[{parent_ref_key}] (Document: {file_name_for_ref})\n\n"
            ref_mapping[parent_ref_key] = { # Use string key for parent refs
                'url': web_url_for_ref,
                'file_name': file_name_for_ref,
                'pages': [], # No pages for parent doc itself
                'doc_type': 'parent_document'
            }
            ref_idx_counter += 1

            # Collect all annexes from this parent document
            for annex in p_doc.get('related_annexes', []):
                all_annexes_for_response[annex['name']] = annex 

        # Now, add Chunk Documents as references
        for result in search_results:
            content = result.get('content', '')
            parent_id_of_chunk = result.get('parent_id')
            
            # Retrieve file_name and web_url from the already retrieved parent_documents
            parent_doc_for_chunk = parent_documents.get(parent_id_of_chunk, {})
            file_name_of_chunk = parent_doc_for_chunk.get('file_name', 'unknown')
            web_url_of_chunk = parent_doc_for_chunk.get('web_url', '')

            pages = result.get('pages', [])

            # Truncate content of chunks to keep context shorter, if needed
            # For now, let's keep it full as search_results are already top_k
            # truncated_content = content[:1000] + '...' if len(content) > 1000 else content

            # Use a unique reference key for chunks, e.g., 'C1', 'C2'
            chunk_ref_key = f"C{ref_idx_counter}"
            context_with_refs += f"[{chunk_ref_key}] (from: {file_name_of_chunk})\\n{content}\\n\\n" # Use full content for chunks
            ref_mapping[chunk_ref_key] = { # Keep integer keys for chunk refs
                'url': web_url_of_chunk, # URL of the main document
                'file_name': file_name_of_chunk,
                'pages': pages,
                'doc_type': 'chunk',
                'parent_id': parent_id_of_chunk
            }
            ref_idx_counter += 1

        # Step 3: Call GPT with function calling
        # Use the English-only system_prompt as agreed
        system_prompt = f"""Meeting Summarization AI Assistant

        Role Description:
        You are an AI assistant for meeting summarization. Your primary role is to process and summarize meeting content accurately, concisely, and relevantly, using ONLY the provided information.

        Operating Guidelines (Strictly Adhere):
        Respond ONLY in English. Regardless of the user's query language, all responses must be in English.
        Answer directly: Only answer what the user asks. Do not provide additional information that is not requested.
        Use ONLY provided information: All answers must be based on the provided meeting information or related documents. DO NOT use external knowledge.
        Concise and clear: Summarize the key points as briefly as possible and use easy-to-understand language. If there are numbers or quantities, include them every time for clarity. If the content and answer are too long, summarize only the key points.

        Always cite sources: Every time you respond, you MUST attach the document name and a clickable link (referencing the provided meeting information or supporting documents) so users can verify the original data. **If an annex or attachment is mentioned in the response content, also attach its link.**

        Summarization Process:
        Receive meeting data: Process meeting information provided in text format (e.g., transcripts, notes, or supporting documents).
        Identify key issues: Find and distinguish main topics, debated issues, important information, and recommendations.
        Summarize resolutions and decisions: Clearly state meeting resolutions, agreements, and decisions made during discussions.
        Define Action Items: Specify tasks (What), responsible persons (Who), and deadlines (When) if available in the original data.
        Prepare meeting summary report: Create a concise, easy-to-understand, well-structured meeting summary report using business professional language.

        If no information is found:
        If no answer is found in the provided information, respond with:
        "I apologize, but I could not find the information you requested. 🙅‍♀️ If you have more details or would like to search for another topic, please let me know. ✍️"

        Response Format:
        The summarized answer: Always prefix with 📄.

        Recommended output format for meeting summary reports:
        Meeting Summary: [Meeting Title]
        Date: [Meeting Date]
        Time: [Time covered by summary]
        Attendees (if available): [List of attendees]

        Main Discussion Points:
        [Point 1]
        [Point 2]
        [Sub-details]

        Resolutions and Decisions:
        [Resolution/Decision 1]
        [Resolution/Decision 2]

        Action Items:
        Task (What)              Responsible (Who)              Due Date (When)
        [Task 1]                 [Person/Team]                  [Date]
        [Task 2]                 [Person/Team]                  [Date]

        References 📚: All relevant referenced documents must be attached to the summarized content. If there are multiple referenced documents, prioritize the main document. (Only shown if an answer can be found). **Also, attach links to annexes/attachments if they are mentioned in the response content. Display in the format: [Annex Name](Annex_URL) after the main references.**
        """

        user_prompt = f"""Documents:
        {context_with_refs}

        Question: {query}

        Answer:"""

        functions = [
            {
                "name": "answer_with_citations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reply": {
                            "type": "string",
                            "description": "Answer from documents" # Changed description
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Reference keys used in the answer, e.g., [\"P1\", \"C3\"]" # Changed description
                        },
                        "mentioned_annexes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Names of annexes/attachments mentioned in the reply" # Changed description
                        }
                    },
                    "required": ["reply", "citations"]
                }
            }
        ]
        
        try:
            response = gpt_client.chat.completions.create(
                model=GPT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                functions=functions,
                function_call={"name": "answer_with_citations"},
                temperature=0.1,
                max_tokens=1000 # Increased max_tokens as response might be longer with annexes
            )
            # Extract token usage from response
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            # Calculate cost (USD)
            pricing_key = "gpt-4o-mini"
            if "gpt-4o" in GPT_DEPLOYMENT.lower() and "mini" not in GPT_DEPLOYMENT.lower():
                pricing_key = "gpt-4o"
            pricing = TOKEN_PRICING.get(pricing_key, TOKEN_PRICING["gpt-4o-mini"])
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            total_cost_usd = input_cost + output_cost
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"status": "Failed", "statusDetail": f"GPT call failed: {str(e)}"}),
                mimetype="application/json", status_code=500
            )
        
        # Step 4: Process response and add citations
        function_call = response.choices[0].message.function_call
        if function_call and function_call.name == "answer_with_citations":
            result = json.loads(function_call.arguments)
            reply = result.get("reply", "")
            citations = result.get("citations", [])
            mentioned_annexes = result.get("mentioned_annexes", [])

            unique_citations = sorted(list(set(citations)))
            citation_links = []
            
            # Build citation links for main documents (chunks or parent docs)
            for ref_key in unique_citations:
                if ref_key in ref_mapping:
                    ref_info = ref_mapping[ref_key]
                    url = ref_info.get('url', '') 
                    file_name = ref_info.get('file_name', 'unknown')
                    pages = ref_info.get('pages', [])
                    
                    if pages and len(pages) > 0:
                        sorted_pages = sorted(pages)
                        if len(sorted_pages) == 1:
                            display_name = f"{file_name} (page {sorted_pages[0]})"
                        else:
                            display_name = f"{file_name} (pages {sorted_pages[0]}-{sorted_pages[-1]})"
                    else:
                        display_name = file_name # For parent docs or whole document
                    
                    # Ensure URL is not empty before creating the link
                    if url:
                        citation_links.append(f"[{display_name}]({url})")
                    else:
                        citation_links.append(f"{display_name} (URL not available)")
            
            # Add annex citations if they were mentioned by the LLM
            for mentioned_annex_name in mentioned_annexes:
                if mentioned_annex_name in all_annexes_for_response:
                    annex_data = all_annexes_for_response[mentioned_annex_name]
                    if annex_data.get('url'):
                        citation_links.append(f"[{mentioned_annex_name}]({annex_data['url']})")

            # Append citations to reply
            if citation_links:
                reply += "\\n\\n**References 📚:** " + ", ".join(citation_links) # Revert to thai (แหล่งอ้างอิง) or English (References) for final
        else:
            reply = response.choices[0].message.content or "Could not generate a response."

        # Save usage to usages container
        try:
            save_usage_to_cosmos(
                database_name=database_body,
                operation="generate",
                model=GPT_DEPLOYMENT,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=total_cost_usd,
                input_text=query,
                output_text=reply,
                additional_data={
                    "top_k": top_k,
                    "container": container_body,
                    "search_result_count": len(search_results)
                }
            )
        except Exception as e:
            logging.error(f"Failed to save usage data for generate operation: {e}")
        
        logging.info(f"✓ Generated response for query: {query}")
        return func.HttpResponse(
            json.dumps({
                "status": "Completed",
                "reply": reply,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": round(total_cost_usd, 8)
                }
            }),
            mimetype="application/json", status_code=200
        )
        
    except ValueError:
        return func.HttpResponse(
            json.dumps({"status": "Failed", "statusDetail": "Invalid JSON format in request"}),
            mimetype="application/json", status_code=400
        )
    except Exception as e:
        error_message = str(e)
        logging.error(f"Unexpected error in generate function: {error_message}")
        return func.HttpResponse(
            json.dumps({"status": "Failed", "statusDetail": f"Unexpected error: {error_message}"}),
            mimetype="application/json", status_code=500
        )
# --- End Azure Function: generate ---


# --- Azure Function: test (no changes) ---
@app.route(route="test")
def test(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')
    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
# --- End Azure Function: test ---
