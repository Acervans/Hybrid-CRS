import sys
import httpx
import json

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import ChatMessage

sys.path.append("..")  # For modular development

OLLAMA_API_URL = "http://127.0.0.1:11434/api"
REQUEST_TIMEOUT = 120.0

Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=REQUEST_TIMEOUT)


### UTILITY FUNCTIONS ###


async def ollama_api_proxy(
    method: str, endpoint: str, request: Request, response: Response
):
    # Reverse proxy for Ollama API
    url: str = f"{OLLAMA_API_URL}/{endpoint}"
    body: bytes = await request.body()

    async def streaming_response():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                method=method,
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
                    method=method,
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


@app.get("/chat")
async def stream_llm():
    gen = (
        response.delta
        for response in Settings.llm.stream_chat(
            [ChatMessage(role="user", content="Hello")]
        )
    )
    return StreamingResponse(gen)


@app.get("/ollama/api/{endpoint}")
async def ollama_api_get(endpoint: str, request: Request, response: Response):
    return await ollama_api_proxy("GET", endpoint, request, response)


@app.post("/ollama/api/{endpoint}")
async def ollama_api_post(endpoint: str, request: Request, response: Response):
    return await ollama_api_proxy("POST", endpoint, request, response)


@app.delete("/ollama/api/{endpoint}")
async def ollama_api_post(endpoint: str, request: Request, response: Response):
    return await ollama_api_proxy("DELETE", endpoint, request, response)
