""" Request schemas for api.py endpoints.
"""

from fastapi import UploadFile
from pydantic import BaseModel
from typing import List, Optional


class InferColumnRolesRequest(BaseModel):
    column_names: List[str]
    file_type: str


class InferFromSampleRequest(BaseModel):
    sample_values: List[str]


class Column(BaseModel):
    id: int
    name: str
    role: str
    data_type: str
    delimiter: Optional[str]
    original_name: str


class SniffResult(BaseModel):
    delimiter: str
    has_header: bool
    labels: List[str]
    newline_str: str
    quote_char: Optional[str]
    total_rows: int


class DatasetFile(BaseModel):
    id: str
    name: str
    original_name: str
    file: Optional[UploadFile]
    file_type: str
    headers: Optional[List[str]]
    columns: List[Column]
    sniff_result: SniffResult


class AgentConfig(BaseModel):
    agent_name: str
    dataset_name: str
    description: Optional[str]
    public: bool


class AgentRequest(BaseModel):
    agent_id: int
    dataset_name: str
    user_id: str


class ChatHistoryRequest(BaseModel):
    chat_id: int
    user_id: str


class CreateChatHistoryRequest(ChatHistoryRequest):
    content: str


class AppendChatHistoryRequest(ChatHistoryRequest):
    new_message: str


class StartWorkflowRequest(AgentRequest):
    agent_name: str


class SendUserResponseRequest(BaseModel):
    workflow_id: str
    user_response: str
