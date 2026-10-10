
import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_prediction(client):
    sample = {
        "albedo": 0.6,
        "SST": 275,
        "U": 5,
        "V": -2,
        "SWD": 200
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    assert "predicted_sic" in response.json()
    assert isinstance(
        response.json()["predicted_sic"],
        (int, float)
    )


def test_invalid_albedo(client):
    sample = {
        "albedo": 1.5,
        "SST": 275,
        "U": 5,
        "V": -2,
        "SWD": 200
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 422

def test_model_unavailable(client, monkeypatch):
    import app as api_module

    monkeypatch.setattr(api_module, "model", None)

    response = client.post(
        "/predict",
        json={
            "albedo": 0.6,
            "SST": 275,
            "U": 5,
            "V": -2,
            "SWD": 200
        }
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model unavailable"

