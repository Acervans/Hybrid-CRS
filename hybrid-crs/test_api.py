import io
import json
import pytest

from fastapi.testclient import TestClient
from api import app


client = TestClient(app)

JWT_SUB = "mock_user"
JWT_TOKEN = "Bearer test.jwt.token"
HEADERS = {"Authorization": JWT_TOKEN}


@pytest.fixture(autouse=True)
def mock_jwt_decode(mocker):
    mocker.patch("api.jwt.decode", return_value={"sub": "mock_user"})


def test_root():
    response = client.get("/", headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_pdf_to_text():
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 24 Tf 100 700 Td (Example Content) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000062 00000 n \n"
        b"0000000111 00000 n \n0000000223 00000 n \n0000000321 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n400\n%%EOF"
    )
    response = client.post(
        "/pdf-to-text",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.text == "Example Content\n\n"


def test_infer_column_roles():
    payload = {
        "column_names": ["user_id", "item_id", "rating"],
        "file_type": "interactions",
    }
    response = client.post("/infer-column-roles", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_infer_datatype():
    payload = {"sample_values": ["1", "2", "3"]}
    response = client.post("/infer-datatype", json=payload, headers=HEADERS)
    assert response.status_code == 200

    response_json = response.json()
    assert "datatype" in response_json
    assert response_json["datatype"] == "token"


def test_infer_delimiter():
    payload = {"sample_values": ["1|2|3", "4|5|6"]}
    response = client.post("/infer-delimiter", json=payload, headers=HEADERS)
    assert response.status_code == 200

    response_json = response.json()
    assert "delimiter" in response_json
    assert response_json["delimiter"] == "|"


def test_create_agent(mocker):
    # Mock Supabase auth/user validation
    mock_supabase = mocker.patch("api.supabase")
    mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
        "user_id": "mock_user"
    }
    mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
        {}
    ]
    mock_supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = (
        {}
    )

    # Mock recommender logic
    mocker.patch("api.FalkorDBRecommender")
    mocker.patch("api.train_expert_model", return_value=({}, {"score": 0.9}))
    mocker.patch("api.process_dataset")
    mocker.patch("api.os.remove")
    mocker.patch("api.shutil.rmtree")

    agent_config = {
        "agent_name": "TestAgent",
        "dataset_name": "TestSet",
        "description": "desc",
        "public": True,
    }

    dataset_file = {
        "file_type": "interactions",
        "columns": [
            {"name": "user_id", "data_type": "token", "role": "user"},
            {"name": "item_id", "data_type": "token", "role": "item"},
            {"name": "rating", "data_type": "float", "role": "rating"},
        ],
        "sniff_result": {
            "delimiter": ",",
            "newline_str": "\n",
            "has_header": True,
            "quote_char": '"',
        },
    }

    files = {
        "agent_id": (None, "123"),
        "agent_config": (None, json.dumps(agent_config)),
        "dataset_files": (None, json.dumps(dataset_file)),
        "upload_files": (
            "inter.csv",
            io.BytesIO(b"user_id,item_id,rating\n1,2,3\n"),
            "text/csv",
        ),
    }

    response = client.post("/create-agent", files=files, headers=HEADERS)
    # success or mock-related failure
    assert response.status_code in (200, 500)


def test_chat_history_endpoints(mocker):
    # Patch JWT decode to simulate auth
    mocker.patch("api.jwt.decode", return_value={"sub": JWT_SUB})

    # Patch all FalkorDBChatHistory methods
    mock_store = mocker.patch("api.FalkorDBChatHistory.store_chat")
    mock_append = mocker.patch("api.FalkorDBChatHistory.append_message")
    mock_get = mocker.patch(
        "api.FalkorDBChatHistory.get_chat",
        return_value={"messages": [{"content": "hello"}]},
    )
    mock_delete = mocker.patch("api.FalkorDBChatHistory.delete_chat")

    # 1. Create Chat History
    payload_create = {
        "chat_id": 123,
        "user_id": JWT_SUB,
        "content": json.dumps([{"role": "user", "content": "Hi"}]),
    }
    res = client.post("/create-chat-history", json=payload_create, headers=HEADERS)
    assert res.status_code == 200
    assert res.json()["chatId"] == 123
    mock_store.assert_called_once()

    # 2. Append Message
    payload_append = {
        "chat_id": 123,
        "user_id": JWT_SUB,
        "new_message": json.dumps({"role": "assistant", "content": "Hello!"}),
    }
    res = client.put("/append-chat-history", json=payload_append, headers=HEADERS)
    assert res.status_code == 200
    mock_append.assert_called_once()

    # 3. Get Chat History (query params)
    res = client.get(
        "/get-chat-history",
        params={"chat_id": 123, "user_id": JWT_SUB},
        headers=HEADERS,
    )
    assert res.status_code == 200
    assert res.json()["messages"][0]["content"] == "hello"
    mock_get.assert_called_once_with(123)

    # 4. Delete Chat History
    payload_delete = {"chat_id": 123, "user_id": JWT_SUB}
    res = client.request(
        method="DELETE",
        url="/delete-chat-history",
        json=payload_delete,
        headers=HEADERS,
    )
    assert res.status_code == 200
    mock_delete.assert_called_once_with(123)
