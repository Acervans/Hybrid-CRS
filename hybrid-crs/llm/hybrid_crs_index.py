from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor  # Metadata extractors using LLM
from llama_index.readers.file import PandasCSVReader

from llama_index.core.ingestion import IngestionPipeline, IngestionCache

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

CTX_WINDOW = 16384

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

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
    port=6333
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
)

# Set up the QdrantVectorStore and StorageContext
vector_store = QdrantVectorStore(client=client, collection_name="ml-small")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if not client.collection_exists("ml-small"):
    # Load documents
    reader = PandasCSVReader(concat_rows=True)
    documents = reader.load_data("../recsys/datasets/ml-latest-small/movies.csv")

    # Create IngestionPipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(separator="\n", chunk_size=768), # 1024
            Settings.embed_model,
        ],
        vector_store=vector_store
    )

    # Run the pipeline to:
    #   1. Turn documents into nodes with embeddings
    #   2. Add nodes to the Qdrant collection
    nodes = pipeline.run(documents=documents, show_progress=True)
    print("Added documents to Qdrant collection.\n")

# Create index from Qdrant vector store
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

if __name__ == '__main__':

    chat_engine = index.as_chat_engine(chat_mode=ChatMode.BEST)

    response_stream = chat_engine.stream_chat("Recommend 3 comedy movies between the years 2010 and 2015")
    response_stream.print_response_stream()
