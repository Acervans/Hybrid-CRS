""" API for HybridCRS. FastAPI backend for HybridCRS that powers LLM-based chat (Ollama),
    web search, PDF parsing, agent workflows, and column inference for recommender systems.
"""

import os
import shutil
import jwt
import httpx
import json
import pymupdf
import html2text
import asyncio
import dotenv
import logging
import uuid
import traceback
import fireducks.pandas as pd

from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from falkordb import FalkorDB
from redis.exceptions import ResponseError

from supabase import create_client, Client

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import HumanResponseEvent

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from io import StringIO

from schemas import (
    AddInteractionsRequest,
    AgentConfig,
    ChatHistoryRequest,
    AppendChatHistoryRequest,
    CreateChatHistoryRequest,
    DatasetFile,
    InferColumnRolesRequest,
    InferFromSampleRequest,
    AgentRequest,
    StartWorkflowRequest,
    SendUserResponseRequest,
)

from llm.hybrid_crs_workflow import HybridCRSWorkflow, StreamEvent, update_dataset
from llm.falkordb_chat_history import FalkorDBChatHistory
from recsys.falkordb_recommender import FalkorDBRecommender
from recsys.recbole_utils import (
    hyperparam_grid_search,
    run_recbole,
    parse_model,
    load_data_and_model,
)
from data_processing.data_utils import (
    InterHeaders,
    UserHeaders,
    ItemHeaders,
    get_datatype,
    get_inter_headers,
    get_item_headers,
    get_user_headers,
    sniff_delimiter,
    process_dataset,
    clean_dataframe,
    SEP,
    QUOTE_MINIMAL,
)

dotenv.load_dotenv()

html2text.config.IMAGES_TO_ALT = True
html2text.config.BODY_WIDTH = 0

HOSTNAME = os.getenv("LOCAL_SERVICES_HOST", "localhost")

OLLAMA_API_URL = f"http://{HOSTNAME}:11434"
OLLAMA_API_PROXY = "/ollama/api/{endpoint}"
OLLAMA_API_EXAMPLES = {
    "generate": {
        "summary": "Generate Completion",
        "description": "Example LLM generation, for endpoint 'generate'.",
        "value": {
            "model": "qwen2.5:3b",
            "prompt": "How are you today?",
            "stream": False,
        },
    },
    "chat": {
        "summary": "Generate Chat Completion",
        "description": "Example LLM chat generation, for endpoint 'chat'.",
        "value": {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": "Why is the sky blue?"}],
            "stream": False,
        },
    },
}

REQUEST_TIMEOUT = 240.0

WEB_SEARCH_TIMEOUT = 10
WEB_SEARCH_RESULTS = 2

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")

EXPERT_MODEL = "EASE"
HYPERPARAM_GRID = {"reg_weight": [1.0, 10.0, 100.0, 250.0, 500.0, 750.0, 1000.0]}

Settings.llm = Ollama(
    model="qwen2.5:3b",
    base_url=OLLAMA_API_URL,
    request_timeout=REQUEST_TIMEOUT,
)

# Setup logging functionality

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


### UTILITY FUNCTIONS ###


async def scrape_search_result(
    result: dict, rank: int, client: httpx.AsyncClient, timeout=WEB_SEARCH_TIMEOUT
) -> dict | None:
    """Scrapes the website content of a search `result`

    - Args:
        - result (dict): Dictionary with search result data
        - rank (int): Rank of the search result
        - client (httpx.AsyncClient): HTTP client
        - timeout (float): Request timeout

    - Returns:
        - dict: result data with scraped content
        - None: if exception occurs
    """
    try:
        response = await client.get(result["href"], timeout=timeout)
    except (httpx.TimeoutException, httpx.RemoteProtocolError):
        return None
    return {
        "rank": rank,
        "title": result["title"],
        "description": result["body"],
        "url": result["href"],
        "content": html2text.html2text(response.text),
    }


async def web_search(
    query: str, timeout=WEB_SEARCH_TIMEOUT, max_results=WEB_SEARCH_RESULTS
) -> list[dict]:
    """Searches the web for `query`

    - Args:
        - query (str): Search query

    - Returns:
        - list[dict]: search results
    """
    ddgs = DDGS(timeout=timeout)
    try:
        results = ddgs.text(query, max_results=max_results, safesearch="moderate")
        async with httpx.AsyncClient() as client:
            tasks = [
                scrape_search_result(result, rank, client, timeout)
                for rank, result in enumerate(results, 1)
            ]
            final_results = sorted(
                filter(None, await asyncio.gather(*tasks)), key=lambda x: x["rank"]
            )

            return final_results
    except DuckDuckGoSearchException:
        return ["Failed to search the web"]


def get_dataset_name(original_name: str, agent_id: int) -> str:
    """Gets unique dataset name

    - Args:
        - original_name (str): Original name of the dataset in Supabase
        - agent_id (int): Agent ID

    - Returns:
        - str: unique dataset name
    """
    return f"{"-".join(original_name.split()).lower()}-{agent_id}"


def get_dataset_path(dataset_name: str, processed: bool = False) -> str:
    """Gets dataset path from unique dataset name

    - Args:
        - dataset_name (str): Name of the dataset
        - processed (bool, optional): Whether to get processed or raw path

    Returns:
        - str: path of an agents' dataset files
    """
    return f"./data_processing/datasets/{'processed' if processed else 'raw'}/{dataset_name}"


def train_expert_model(
    dataset_name: str,
    model: str = EXPERT_MODEL,
    param_grid: dict = HYPERPARAM_GRID,
    delete_on_error: bool = True,
) -> tuple[dict, dict]:
    """Trains expert model using dataset_name with hyperparameter tuning

    - Args:
        - dataset_name (str): Name of the dataset
        - model (str): Model to train
        - param_grid (dict): Parameters to combine for tuning
        - delete_on_error (bool): Whether to delete model and dataset on error

    Returns:
        - tuple[dict, dict]: best parameters and test scores
    """
    dataset_path = f"recsys/saved/{dataset_name}-Dataset.pth"
    model_path = f"recsys/saved/{dataset_name}.pth"
    try:
        best_params, test_scores = hyperparam_grid_search(
            model=model,
            param_grid=param_grid,
            config_file="recsys/config/generic.yaml",
            config_dict={
                "save_dataset": True,
                "dataset_save_path": dataset_path,
                "data_path": "./data_processing/datasets/processed",
                "state": "CRITICAL",
            },
            dataset_name=dataset_name,
            save_best_model_path=model_path,
            tensorboard_log_dir="recsys",
        )
    except Exception as e:
        if delete_on_error:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
        raise e
    return best_params, test_scores


### API DEFINITION ###

app = FastAPI(
    title="HybridCRS API",
    summary="Application Programming Interface for the HybridCRS Platform",
)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize FalkorDB client
db = FalkorDB(host=HOSTNAME, port=6379)


# Swagger API Docs Auth
def auth_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="1.0",
        summary=app.summary,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
    }
    openapi_schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = auth_openapi

# Store workflow instances in a dictionary
workflows = {}

allowed_origins = [
    f"http://localhost:3000",  # Dev
    f"http://localhost:3001",  # Dev
    "https://hybrid-crs.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.middleware("http")
async def verify_jwt(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in ("/docs", "/openapi.json"):
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse("Missing authorization token", status_code=401)

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(
            jwt=token,
            key=SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        request.state.jwt = payload

        response = await call_next(request)
        return response
    except jwt.ExpiredSignatureError:
        return JSONResponse("Token expired", status_code=401)
    except (jwt.InvalidSignatureError, jwt.InvalidTokenError, jwt.DecodeError) as e:
        return JSONResponse(f"Access denied: {e}", status_code=403)


@app.get(OLLAMA_API_PROXY, openapi_extra={"requestBody": None})
@app.post(OLLAMA_API_PROXY)
@app.delete(OLLAMA_API_PROXY)
async def ollama_api_proxy(
    endpoint: str,
    request: Request,
    response: Response,
    body: dict = Body(
        default={},
        description=(
            "Body of the request. Visit the [Ollama API documentation]"
            "(https://github.com/ollama/ollama/blob/main/docs/api.md) for reference."
        ),
        media_type="application/json",
        openapi_examples=OLLAMA_API_EXAMPLES,
    ),
):
    """Proxy for Ollama API requests

    - Args:
        - endpoint (str): Ollama API endpoint
        - request (Request): Request object with request data
        - response (Response): Response object with partial response
        - body (dict): Body of the request if needed

    - Returns:
        - StreamingResponse | Response: response to Ollama API call
    """
    # Reverse proxy for Ollama API
    url: str = f"{OLLAMA_API_URL}/api/{endpoint}"

    # Perform Web Search with DuckDuckGo
    if request.headers.get("websearch") == "true":
        web_results = await web_search(body["messages"][-1]["content"])
        body["messages"][-1][
            "content"
        ] += f"\n\nWeb search obtained these results, CITE THE SOURCES: \n{web_results}"

    body_bytes = json.dumps(body).encode("utf-8") if body else None

    async def streaming_response():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                method=request.method,
                url=url,
                data=body_bytes,
                params=request.query_params,
                timeout=REQUEST_TIMEOUT,
            ) as stream_response:
                if stream_response.status_code != 200:
                    response_text = await stream_response.aread()
                    raise HTTPException(
                        status_code=stream_response.status_code,
                        detail=response_text.decode(),
                    )
                async for chunk in stream_response.aiter_bytes():
                    yield chunk

    if len(body) > 0 and body.get("stream", True):
        # Streaming reponse
        return StreamingResponse(streaming_response())
    else:
        # Non-streaming response
        async with httpx.AsyncClient() as client:
            try:
                proxy = await client.request(
                    method=request.method,
                    url=url,
                    data=body_bytes,
                    params=request.query_params,
                    timeout=REQUEST_TIMEOUT,
                )
                response.body = proxy.content
                response.status_code = proxy.status_code
                return response
            except httpx.ReadTimeout:
                raise HTTPException(
                    status_code=500,
                    detail="Request took too long to generate a response",
                )


@app.post("/pdf-to-text")
async def pdf_to_text(file: UploadFile) -> Response:
    """Extracts readable text from a PDF `file`

    - Args:
        - file (UploadFile): PDF file

    - Returns:
        - Response: response with extracted text
    """
    doc = pymupdf.open(stream=file.file.read(), filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text() + "\n"

    return Response(text)


@app.post("/infer-column-roles")
async def infer_column_roles(
    payload: InferColumnRolesRequest = Body(...),
) -> JSONResponse:
    """Infers column roles given names and file type

    - Args:
        - payload (InferColumnRolesRequest): Column names and file type

    - Returns:
        - JSONResponse: response with reverse mapping of inferred roles
    """
    column_names = payload.column_names
    file_type = payload.file_type

    match (file_type):
        case "interactions":
            res = get_inter_headers(column_names)
        case "users":
            res = get_user_headers(column_names)
        case "items":
            res = get_item_headers(column_names)
        case _:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_type} is not supported",
            )
    rev_map = {v: k[:-7] for k, v in res.model_dump().items() if v is not None}
    return JSONResponse(rev_map)


@app.post("/infer-datatype")
async def infer_datatype(payload: InferFromSampleRequest = Body(...)) -> JSONResponse:
    """Infers data type given a sample of values

    - Args:
        - payload (InferFromSampleRequest): Sample values

    - Returns:
        - JSONResponse: response with datatype and delimiter if sequential type
    """
    try:
        datatype = get_datatype(payload.sample_values)
        return JSONResponse(
            {
                "datatype": datatype,
                "delimiter": (
                    sniff_delimiter(payload.sample_values)
                    if datatype.endswith("seq")
                    else None
                ),
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


@app.post("/infer-delimiter")
async def infer_delimiter(payload: InferFromSampleRequest = Body(...)) -> JSONResponse:
    """Detects sequence delimiter given a sample of values

    - Args:
        - payload (InferFromSampleRequest): Sample values

    - Returns:
        - JSONResponse: response with delimiter
    """
    try:
        return JSONResponse({"delimiter": sniff_delimiter(payload.sample_values)})
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


@app.post("/create-agent")
async def create_agent(
    request: Request,
    agent_id: int = Form(...),
    agent_config: str = Form(...),
    dataset_files: list[str] = Form(...),
    upload_files: list[UploadFile] = File(...),
) -> JSONResponse:
    """Creates a recommendation agent with given configuration and dataset files

    - Args:
        - agent_id (int): List of sample values
        - agent_config (str): Agent configuration as JSON string
        - dataset_files (list[str]): List of dataset metadata as JSON strings
        - upload_files (list[UploadFile]): List of dataset files

    - Returns:
        - JSONResponse: response with agent data and model test scores
    """
    try:
        # Agent existence validation
        response = (
            supabase.table("RecommenderAgent")
            .select("*")
            .eq("agent_id", agent_id)
            .single()
            .execute()
        )
        metadata = response.data

        # Author validation
        assert request.state.jwt["sub"] == metadata["user_id"]

        # Request validation
        agent_config_obj: AgentConfig = AgentConfig.model_validate_json(agent_config)
        dataset_files_obj: list[DatasetFile] = [
            DatasetFile.model_validate_json(file) for file in dataset_files
        ]
        for i in range(len(upload_files)):
            dataset_files_obj[i].file = upload_files[i]

        if not agent_config_obj.agent_name.strip():
            raise HTTPException(status_code=400, detail="Agent name is required")
        if not agent_config_obj.dataset_name.strip():
            raise HTTPException(status_code=400, detail="Dataset name is required")

        # Dataset path
        dataset_name = get_dataset_name(agent_config_obj.dataset_name, agent_id)
        dataset_path = get_dataset_path(dataset_name)
        output_path = get_dataset_path(dataset_name, True)

        shutil.rmtree(dataset_path, ignore_errors=True)
        os.makedirs(dataset_path)

        try:
            dataset_roles = {}

            # Process dataset files
            for file_obj in dataset_files_obj:
                headers = []
                seq_col_delim = {}
                roles_dict = {}

                for i in range(len(file_obj.columns)):
                    column = file_obj.columns[i]
                    column.name = column.name.replace(":", "_").replace(" ", "_")
                    processed_column_name = f"{column.name}:{column.data_type}"
                    headers.append(processed_column_name)

                    if column.data_type.endswith("seq"):
                        seq_col_delim[i] = column.delimiter

                    if column.role != "extra":
                        roles_dict[f"{column.role}_column"] = processed_column_name

                # Role headers + File extension by type
                if file_obj.file_type == "interactions":
                    roles_dict = InterHeaders.model_validate(roles_dict)
                    ext = "inter"
                elif file_obj.file_type == "users":
                    roles_dict = UserHeaders.model_validate(roles_dict)
                    ext = "user"
                else:
                    roles_dict = ItemHeaders.model_validate(roles_dict)
                    ext = "item"

                dataset_roles[file_obj.file_type] = roles_dict

                # Parse contents as CSV
                sniff = file_obj.sniff_result
                df = pd.read_csv(
                    StringIO(
                        initial_value=(
                            (await file_obj.file.read()).decode(
                                "utf-8", errors="replace"
                            )
                        ),
                        newline=sniff.newline_str,
                    ),
                    delimiter=sniff.delimiter,
                    quotechar=sniff.quote_char or '"',
                    lineterminator=sniff.newline_str,
                    skiprows=1 if sniff.has_header else None,
                    names=headers,
                    on_bad_lines="warn",
                )

                # Normalize `*_seq` columns
                for idx, orig_delim in seq_col_delim.items():
                    col = headers[idx]
                    if orig_delim != " ":
                        df[col] = (
                            df[col]
                            .str.replace(" ", "-", regex=False)
                            .replace(orig_delim, " ", regex=False)
                        )

                # Save raw dataset file
                save_path = f"{dataset_path}/{dataset_name}.{ext}"
                df.to_csv(
                    save_path,
                    sep=sniff.delimiter,
                    quotechar=sniff.quote_char or '"',
                    lineterminator=sniff.newline_str,
                    index=False,
                    header=True,
                )

                # Free some memory
                del df

            # Process and clean dataset files
            process_dataset(
                dataset_name=dataset_name,
                dataset_dir=dataset_path,
                output_dir=output_path,
                user_headers=dataset_roles.get("users"),
                item_headers=dataset_roles.get("items"),
                inter_headers=dataset_roles.get("interactions"),
            )

            # Start FalkorDB graph training
            FalkorDBRecommender(
                dataset_name=dataset_name, dataset_dir=output_path, db=db, clear=True
            )

            # Start expert model training
            _, test_scores = train_expert_model(
                dataset_name=dataset_name,
                model=EXPERT_MODEL,
                param_grid=HYPERPARAM_GRID,
                delete_on_error=True,
            )

        except Exception as e:
            shutil.rmtree(dataset_path, ignore_errors=True)
            shutil.rmtree(output_path, ignore_errors=True)
            try:
                db.select_graph(dataset_name).delete()
            except ResponseError:
                pass
            raise e

        # Delete unprocessed dataset files
        shutil.rmtree(dataset_path, ignore_errors=True)

        response = (
            supabase.table("RecommenderAgent")
            .update({"processed": True})
            .eq("agent_id", agent_id)
            .execute()
        ).data[0]

        return JSONResponse(
            {"agentId": agent_id, "agentRow": response, "testScores": test_scores}
        )

    except Exception as e:
        # Delete from Supabase
        supabase.table("RecommenderAgent").delete().eq("agent_id", agent_id).execute()

        print("Error creating agent:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error creating agent: {e}")


@app.delete("/delete-agent")
async def delete_agent(request: Request, payload: AgentRequest = Body(...)):
    """Deletes a recommendation agent given agent ID and dataset name

    - Args:
        - payload (AgentRequest): Agent ID, dataset name & user ID
    """
    try:
        # Author validation
        assert request.state.jwt["sub"] == payload.user_id

        agent_id = payload.agent_id

        # Delete dataset
        dataset_name = get_dataset_name(payload.dataset_name, agent_id)
        proc_dataset_path = get_dataset_path(dataset_name, True)
        raw_dataset_path = get_dataset_path(dataset_name, False)
        shutil.rmtree(proc_dataset_path, ignore_errors=True)
        shutil.rmtree(raw_dataset_path, ignore_errors=True)

        # Delete graph
        db.select_graph(dataset_name).delete()

        # Delete model and model dataset
        model_path = f"recsys/saved/{dataset_name}.pth"
        dataset_path = f"recsys/saved/{dataset_name}-Dataset.pth"
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {e}")


@app.post("/retrain-agent")
async def retrain_agent(request: Request, payload: AgentRequest = Body(...)):
    """Retrains a recommendation agent given agent ID and dataset name

    - Args:
        - payload (AgentRequest): Agent ID, dataset name & user ID

    - Returns:
        - JSONResponse: response with agent data and model test scores
    """
    try:
        # Author validation
        assert request.state.jwt["sub"] == payload.user_id

        agent_id = payload.agent_id
        dataset_name = get_dataset_name(payload.dataset_name, agent_id)
        dataset_path = get_dataset_path(dataset_name, True)

        # Remove previous saved dataset
        model_dataset_path = f"recsys/saved/{dataset_name}-Dataset.pth"
        if os.path.exists(model_dataset_path):
            os.remove(model_dataset_path)

        # Clean dataset files
        inter_dataset = f"{dataset_path}/{dataset_name}.inter"
        user_dataset = f"{dataset_path}/{dataset_name}.user"
        clean_dataframe(pd.read_csv(inter_dataset, sep=SEP), verbose=False).to_csv(
            inter_dataset, sep=SEP, index=False, quoting=QUOTE_MINIMAL, escapechar="\\"
        )
        if os.path.exists(user_dataset):
            clean_dataframe(pd.read_csv(user_dataset, sep=SEP), verbose=False).to_csv(
                user_dataset,
                sep=SEP,
                index=False,
                quoting=QUOTE_MINIMAL,
                escapechar="\\",
            )

        # Retrain expert model
        _, test_scores = train_expert_model(
            dataset_name=dataset_name,
            model=EXPERT_MODEL,
            param_grid=HYPERPARAM_GRID,
            delete_on_error=False,
        )

        # Update processed and new_sessions
        response = (
            supabase.table("RecommenderAgent")
            .update({"processed": True, "new_sessions": 0})
            .eq("agent_id", agent_id)
            .execute()
        ).data[0]

        return JSONResponse(
            {"agentId": agent_id, "agentRow": response, "testScores": test_scores}
        )

    except Exception as e:
        # Cancel processing
        (
            supabase.table("RecommenderAgent")
            .update({"processed": True})
            .eq("agent_id", agent_id)
            .execute()
        )
        raise HTTPException(status_code=500, detail=f"Error retraining agent: {e}")


@app.post("/create-chat-history")
async def create_chat_history(
    request: Request,
    payload: CreateChatHistoryRequest = Body(...),
) -> JSONResponse:
    """Creates a Chat History in the FalkorDB graph store

    Args:
        payload (CreateChatHistoryRequest): Chat ID, user ID and content

    Returns:
        JSONResponse: created Chat History
    """
    try:
        # Author validation
        assert request.state.jwt["sub"] == payload.user_id

        chat_id = payload.chat_id
        content = json.loads(payload.content)

        ch = FalkorDBChatHistory(db=db)
        ch.store_chat(chat_id, content)
        return JSONResponse({"chatId": chat_id, "content": content})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating chat history: {e}")


@app.put("/append-chat-history")
async def append_chat_history(
    request: Request,
    payload: AppendChatHistoryRequest = Body(...),
):
    """Appends a message to an existing Chat History in the FalkorDB graph store

    Args:
        payload (AppendChatHistoryRequest): Chat ID, user ID and new message to append
    """
    try:
        # Author validation
        assert request.state.jwt["sub"] == payload.user_id

        chat_id = payload.chat_id
        new_message = payload.new_message

        ch = FalkorDBChatHistory(db=db)
        ch.append_message(chat_id, json.loads(new_message))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error appending to chat history: {e}"
        )


@app.get("/get-chat-history")
async def get_chat_history(
    request: Request, chat_id: int = Query(...), user_id: str = Query(...)
) -> JSONResponse:
    """Gets an existing Chat History's contents from the FalkorDB graph store

    Args:
        chat_id (int): Chat ID
        user_id (str): User ID

    Returns:
        JSONResponse: list of chat messages
    """
    try:
        # Author validation
        assert request.state.jwt["sub"] == user_id

        ch = FalkorDBChatHistory(db=db)
        return JSONResponse(ch.get_chat(chat_id))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving chat history: {e}"
        )


@app.delete("/delete-chat-history")
async def delete_chat_history(
    request: Request,
    payload: ChatHistoryRequest = Body(...),
):
    """Deletes a Chat History from the FalkorDB graph store

    Args:
        payload (ChatHistoryRequest): Chat ID and user ID
    """
    try:
        # Author validation
        assert request.state.jwt["sub"] == payload.user_id

        chat_id = payload.chat_id

        ch = FalkorDBChatHistory(db=db)
        ch.delete_chat(chat_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat history: {e}")


@app.post("/start-workflow")
async def start_workflow(
    payload: StartWorkflowRequest = Body(...),
) -> StreamingResponse:
    """Starts a workflow (conversation) with a recommendation agent

    - Args:
        - payload (StartWorkflowRequest): User ID, agent ID, agent name & dataset name

    - Returns:
        - StreamingResponse: stream of workflow events
    """
    workflow_id = str(uuid.uuid4())
    agent_id = payload.agent_id

    def archive_session():
        # Set chat session as archived
        (
            supabase.table("ChatHistory")
            .update({"archived": True})
            .eq("agent_id", agent_id)
            .execute()
        )

    try:
        # Increment session count
        supabase.rpc("increment_new_sessions", {"agent_id": agent_id})

        dataset_name = get_dataset_name(payload.dataset_name, agent_id)
        dataset_path = get_dataset_path(dataset_name, True)
        wf = HybridCRSWorkflow(
            wid=workflow_id,
            agent_name=payload.agent_name,
            user_id=payload.user_id,
            dataset_name=payload.dataset_name,
            dataset_dir=dataset_path,
            description=payload.description,
            timeout=REQUEST_TIMEOUT,
            verbose=True,
        )

        # Store the workflow instance
        workflows[workflow_id] = {"wf": wf, "handler": None}

        async def event_generator():
            logger.debug(f"event_generator: created workflow {workflow_id}")
            handler = wf.run()

            # Store the workflow handler
            workflows[workflow_id]["handler"] = handler

            logger.debug(f"event_generator: obtained handler {id(handler)}")
            try:
                # Stream events and yield to client
                async for ev in wf.stream_events():
                    is_stream_event = isinstance(ev, StreamEvent)
                    if not is_stream_event:
                        logger.info(f"Sending message to client: {ev}")
                    yield f"{json.dumps({
                        'event': ev.__repr_name__(),
                        'message': ev.model_dump(),
                        "done": not is_stream_event
                        })}\n\n"
                final_result = await handler

                yield f"{json.dumps({'result': final_result})}\n\n"
            except Exception as e:
                error_message = f"Error in workflow: {str(e)}"
                logger.error(error_message)
                yield f"{json.dumps({'event': 'error', 'message': error_message})}\n\n"
            finally:
                # Clean up
                workflows.pop(workflow_id, None)
                archive_session()

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        archive_session()
        raise HTTPException(status_code=500, detail=f"Error starting workflow: {e}")


@app.post("/send-user-response")
async def send_user_response(
    payload: SendUserResponseRequest = Body(...),
) -> JSONResponse:
    """Sends a user response to an ongoing workflow

    - Args:
        - payload (SendUserResponseRequest): Workflow ID and user response

    - Returns:
        - JSONResponse: response with send status
    """
    workflow_id = payload.workflow_id
    user_response = payload.user_response

    # Get the workflow instance
    wf_dict = workflows.get(workflow_id, {})
    wf = wf_dict.get("wf", None)
    logger.info(f"send_user_response: user response {user_response}")
    if wf:
        # Get workflow handler to send event
        handler = wf_dict.get("handler", None)
        if not handler:
            raise HTTPException(
                status_code=404, detail=f"Handler for workflow {workflow_id} not found"
            )
        handler.ctx.send_event(HumanResponseEvent(response=user_response))
        return JSONResponse(
            {"status": f"Workflow {workflow_id} received: '{user_response}'"}
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow {workflow_id} not found or already completed",
        )


@app.post("/add-interactions")
async def add_interactions(
    payload: AddInteractionsRequest = Body(...),
):
    """Adds interactions to an agent's graph and processed dataset

    Args:
        payload (AddInteractionsRequest): User ID, interactions, agent ID and dataset name
    """
    try:
        user_id = payload.user_id
        item_ids = payload.item_ids
        ratings = payload.ratings

        agent_id = payload.agent_id
        dataset_name = get_dataset_name(payload.dataset_name, agent_id)
        dataset_path = get_dataset_path(dataset_name, True)

        rec = FalkorDBRecommender(
            dataset_name=dataset_name, dataset_dir=dataset_path, db=db, clear=False
        )
        rec.add_user_interactions(user_id, list(zip(item_ids, ratings)))

        inter_path = f"{dataset_path}/{dataset_name}.inter"
        await update_dataset(inter_path, user_id, item_ids, ratings)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error adding user interactions: {e}"
        )
