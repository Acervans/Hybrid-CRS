""" API for HybridCRS. FastAPI backend for HybridCRS that powers LLM-based chat (Ollama),
    web search, PDF parsing, agent workflows, and column inference for recommender systems.
"""

import os
import shutil
import sys
import jwt
import httpx
import json
import pymupdf
import html2text
import asyncio
import dotenv
import logging
import uuid

from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    UploadFile,
)
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import HumanResponseEvent

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from supabase import create_client, Client

from schemas import (
    InferColumnRolesRequest,
    InferFromSampleRequest,
    CreateAgentRequest,
    DeleteAgentRequest,
    StartWorkflowRequest,
    SendUserResponseRequest,
)

from llm.hybrid_crs_workflow import HybridCRSWorkflow, StreamEvent
from data_processing.data_utils import (
    get_datatype,
    get_inter_headers,
    get_item_headers,
    get_user_headers,
    sniff_delimiter,
    normalize,
)

dotenv.load_dotenv()

html2text.config.IMAGES_TO_ALT = True
html2text.config.BODY_WIDTH = 0

OLLAMA_API_URL = "http://127.0.0.1:11434/api"
OLLAMA_API_PROXY = "/ollama/api/{endpoint}"
REQUEST_TIMEOUT = 120.0

WEB_SEARCH_TIMEOUT = 10
WEB_SEARCH_RESULTS = 2

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")

Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=REQUEST_TIMEOUT)


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


### API DEFINITION ###

app = FastAPI(
    title="HybridCRS API",
    summary="Application Programming Interface for the HybridCRS Platform",
)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Store the workflow instances in a dictionary
workflows = {}

# NOTE add frontend deployed url
allowed_origins = [
    "http://localhost:3000",  # Dev
    "http://localhost:3001",  # Dev
    "http://192.168.1.142:3000",
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
        raise HTTPException(status_code=401, detail="Missing authorization token")

    token = auth_header.split(" ")[1]
    try:
        # TODO # Extract user ID/email, restrict access to specific endpoints,
        # restrict DB operations if JWT's user_id and row user_id dont match
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
        raise HTTPException(status_code=401, detail="Token expired")
    except (jwt.InvalidSignatureError, jwt.InvalidTokenError, jwt.DecodeError) as e:
        raise HTTPException(status_code=403, detail=f"Access denied: {e}")


@app.get(OLLAMA_API_PROXY)
@app.post(OLLAMA_API_PROXY)
@app.delete(OLLAMA_API_PROXY)
async def ollama_api_proxy(endpoint: str, request: Request, response: Response):
    """Proxy for Ollama API requests

    - Args:
        - endpoint (str): Ollama API endpoint
        - request (Request): Request object with request data
        - response (Response): Response object with partial response

    - Returns:
        - StreamingResponse | Response: response to Ollama API call
    """
    # Reverse proxy for Ollama API
    url: str = f"{OLLAMA_API_URL}/{endpoint}"
    body: bytes = await request.body()

    # Perform Web Search with DuckDuckGo
    if request.headers.get("websearch") == "true":
        context_body = json.loads(body)
        web_results = await web_search(context_body["messages"][-1]["content"])
        context_body["messages"][-1][
            "content"
        ] += f"\n\nWeb search obtained these results, CITE THE SOURCES: \n{web_results}"
        body = json.dumps(context_body).encode("utf-8")

    async def streaming_response():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                method=request.method,
                url=url,
                data=body,
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

    if len(body) > 0 and json.loads(body).get("stream", True):
        # Streaming reponse
        return StreamingResponse(streaming_response())
    else:
        # Non-streaming response
        async with httpx.AsyncClient() as client:
            try:
                proxy = await client.request(
                    method=request.method,
                    url=url,
                    data=body,
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
async def pdf_to_text(file: UploadFile):
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
async def infer_column_roles(payload: InferColumnRolesRequest = Body(...)):
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
async def infer_datatype(payload: InferFromSampleRequest = Body(...)):
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
async def infer_delimiter(payload: InferFromSampleRequest = Body(...)):
    try:
        return JSONResponse({"delimiter": sniff_delimiter(payload.sample_values)})
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


@app.post("/create-agent")
async def create_agent(payload: CreateAgentRequest = Body(...)):
    # Check if exists in supabase and processing is True. Then create and process files + recommender
    pass


@app.delete("/delete-agent")
async def delete_agent(payload: DeleteAgentRequest = Body(...)):
    pass


@app.post("/start-workflow")
async def start_workflow(payload: StartWorkflowRequest = Body(...)):
    workflow_id = str(uuid.uuid4())
    wf = HybridCRSWorkflow(
        wid=workflow_id,
        user_id=payload.user_id,
        dataset_name=payload.dataset_name,
        timeout=300,
        verbose=True,
    )

    # Store the workflow instance
    workflows[workflow_id] = {"wf": wf, "handler": None}

    async def event_generator():
        logger.debug(f"event_generator: created workflow {workflow_id}")
        yield f"{json.dumps({'workflow_id': workflow_id})}\n\n"

        handler = wf.run()

        # Store the workflow handler
        workflows[workflow_id]["handler"] = handler

        logger.debug(f"event_generator: obtained handler {id(handler)}")
        try:
            # Stream events and yield to client
            async for ev in wf.stream_events():
                if not isinstance(ev, StreamEvent):
                    logger.info(f"Sending message to client: {ev}")
                yield f"{json.dumps({'event': ev.__repr_name__(), 'message': ev.dict()})}\n\n"
            final_result = await handler

            yield f"{json.dumps({'result': final_result})}\n\n"
        except Exception as e:
            error_message = f"Error in workflow: {str(e)}"
            logger.error(error_message)
            yield f"{json.dumps({'event': 'error', 'message': error_message})}\n\n"
        finally:
            # Clean up
            workflows.pop(workflow_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/send-user-response")
async def send_user_response(
    payload: SendUserResponseRequest = Body(...),
) -> JSONResponse:
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
        handler.ctx.send_event(HumanResponseEvent(response=str(user_response)))
        return JSONResponse({"status": "response received"})
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow {workflow_id} not found or already completed",
        )
