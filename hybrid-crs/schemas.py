""" Request schemas for api.py endpoints.
"""

from fastapi import UploadFile
from pydantic import BaseModel


class InferColumnRolesRequest(BaseModel):
    column_names: list[str]
    file_type: str


class InferFromSampleRequest(BaseModel):
    sample_values: list[str]


class CreateAgentRequest(BaseModel):
    id: str
    name: str
    original_name: str
    file: UploadFile
    file_type: str
    headers: list[str] | None
    columns: dict
    sniff_result: dict
    sample_data: list[list[str]]


class DeleteAgentRequest(BaseModel):
    agent_id: str
    user_id: str
    dataset_name: str


class StartWorkflowRequest(BaseModel):
    user_id: str
    dataset_name: str


class SendUserResponseRequest(BaseModel):
    workflow_id: str
    user_response: str
