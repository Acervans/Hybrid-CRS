import sys
import httpx
import json
import pymupdf
import html2text
import asyncio

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

html2text.config.IMAGES_TO_ALT = True
html2text.config.BODY_WIDTH = 0

sys.path.append("..")  # For modular development

OLLAMA_API_URL = "http://127.0.0.1:11434/api"
OLLAMA_API_PROXY = "/ollama/api/{endpoint}"
REQUEST_TIMEOUT = 120.0

WEB_SEARCH_TIMEOUT = 10
WEB_SEARCH_RESULTS = 2

Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=REQUEST_TIMEOUT)


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
