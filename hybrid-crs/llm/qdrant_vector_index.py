from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PandasCSVReader

from llama_index.core.ingestion import IngestionPipeline, IngestionCache

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

CTX_WINDOW = 16384

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Settings.embed_model = HuggingFaceEmbedding(model_name="jxm/cde-small-v1")

# ollama
Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=360.0, context_window=CTX_WINDOW)

# Create a Qdrant client and collection
client = qdrant_client.QdrantClient(
    # use :memory: mode for fast and light-weight experiments,
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    # url="http://<host>:<port>"
    # otherwise set Qdrant instance with host and port:
    host="localhost",
    port=6333,
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
)

# Set up the QdrantVectorStore and StorageContext
vector_store = QdrantVectorStore(client=client, collection_name="sample")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if not client.collection_exists("sample"):
    # Load documents
    documents = SimpleDirectoryReader("../recsys/datasets/samples").load_data()

    # Create IngestionPipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=768),  # 1024
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )

    # Run the pipeline to:
    #   1. Turn documents into nodes with embeddings
    #   2. Add nodes to the Qdrant collection
    nodes = pipeline.run(documents=documents, show_progress=True)
    # pipeline.arun(documents=documents, show_progress=True)  # Maybe get remaining time and show in frontend

    print("Added documents to Qdrant collection.\n")

# Create index from Qdrant vector store
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

if __name__ == "__main__":

    chat_engine = index.as_chat_engine()
    response_stream = chat_engine.stream_chat("What's the moral of the story?")
    response_stream.print_response_stream()
    print("\n", flush=True)

    response_stream = chat_engine.stream_chat("Who are the characters?")
    response_stream.print_response_stream()
    print("\n", flush=True)

    response_stream = chat_engine.stream_chat("Who's the protagonist?")
    response_stream.print_response_stream()
