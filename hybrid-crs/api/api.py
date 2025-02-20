import sys
import httpx
import json
import pymupdf

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

sys.path.append("..")  # For modular development

OLLAMA_API_URL = "http://127.0.0.1:11434/api"
REQUEST_TIMEOUT = 120.0

Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=REQUEST_TIMEOUT)


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


@app.api_route("/ollama/api/{endpoint}", methods=["GET", "POST", "DELETE"])
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
