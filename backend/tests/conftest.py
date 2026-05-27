"""Shared pytest fixtures for backend tests."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="session")
def client():
    """Single TestClient that triggers lifespan startup (loads models once)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_npi(client) -> str:
    """Fetch the top-ranked NPI for use in detail/explain tests."""
    r = client.post("/api/predict", json={"limit": 1})
    return r.json()["physicians"][0]["NPI"]
