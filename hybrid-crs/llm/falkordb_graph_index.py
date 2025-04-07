from llama_index.core import Settings, SimpleDirectoryReader, PropertyGraphIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PandasCSVReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, ChunkNode, Relation

import fireducks.pandas as pd
import nest_asyncio

CTX_WINDOW = 16384

nest_asyncio.apply()

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=360.0, temperature=0.0, context_window=CTX_WINDOW)

# documents = SimpleDirectoryReader("../recsys/datasets/samples").load_data()

# Load documents
reader = PandasCSVReader(concat_rows=False)

# documents = reader.load_data(
#     "../recsys/datasets/ml-latest-small/movies.csv",
# )

items_df = pd.read_csv("../recsys/datasets/ml-latest-small/movies.csv", sep=",")
ratings_df = pd.read_csv("../recsys/datasets/ml-latest-small/ratings.csv", sep=",")
tags_df = pd.read_csv("../recsys/datasets/ml-latest-small/tags.csv", sep=",")

graph_store = FalkorDBPropertyGraphStore(
    url="redis://localhost:6379",
    database="ml-small",
    refresh_schema=True
)

if len(graph_store.get(ids=["u_1"])) == 0:

    # insert nodes
    item_nodes = [
        EntityNode(
            name=f"i_{x.movieId}",
            label="ITEM",
            properties={"item_name": x.title, "genres": x.genres.split("|")},
        )
        for x in items_df.itertuples()
    ]
    graph_store.upsert_nodes(item_nodes)

    user_nodes = [
        EntityNode(
            name=f"u_{x}",
            label="USER"
        )
        for x in ratings_df.userId
    ]
    graph_store.upsert_nodes(user_nodes)

    # insert relationships
    rating_relations = [
        Relation(
            label="RATED",
            source_id=f"u_{x.userId}",
            target_id=f"i_{x.movieId}",
            properties={"rating": x.rating, "timestamp": x.timestamp},
        )
        for x in ratings_df.itertuples()
    ]
    graph_store.upsert_relations(rating_relations)

    feature_relations = [
        Relation(
            label="TAGGED",
            source_id=f"u_{x.userId}",
            target_id=f"i_{x.movieId}",
            properties={"tag": x.tag.lower(), "timestamp": x.timestamp},
        )
        for x in tags_df.itertuples()
    ]
    graph_store.upsert_relations(feature_relations)


# index = PropertyGraphIndex.from_documents(
#     documents,
#     embed_model=Settings.embed_model,
#     kg_extractors=[SchemaLLMPathExtractor(llm=Settings.llm)],
#     property_graph_store=graph_store,
#     show_progress=True,
# )

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    embed_model=Settings.embed_model,
    embed_kg_nodes=True,
    show_progress=True
)

if __name__ == "__main__":

    # chat_engine = index.as_chat_engine()
    # response_stream = chat_engine.stream_chat("What's the moral of the story?")
    # response_stream.print_response_stream()
    # print('\n', flush=True)

    # response_stream = chat_engine.stream_chat("Who are the characters?")
    # response_stream.print_response_stream()
    # print('\n', flush=True)

    # response_stream = chat_engine.stream_chat("Who's the protagonist?")
    # response_stream.print_response_stream()

    # retriever = index.as_retriever(
    #     include_text=True,  # include source text in returned nodes, default True
    # )

    # nodes = retriever.retrieve("What's the moral of the story?")

    # for node in nodes:
    #     print(node.text)

    # query_engine = index.as_query_engine(include_text=True)

    # response = query_engine.query("What's the moral of the story?")

    # print(str(response))

    # retriever = index.as_retriever(
    #     include_text=True,  # include source text in returned nodes, default True
    # )

    # nodes = retriever.retrieve("Recommend me 3 comedy movies between the years 2010 and 2013")

    # for node in nodes:
    #     print(node.text)

    # query_engine = index.as_query_engine(include_text=True)

    # response = query_engine.query(
    #     "Recommend me 3 comedy movies between the years 2010 and 2013"
    # )
    # print(str(response))

    response = index.as_query_engine().query("What comedy movies where made in 2010?")
    print(str(response))