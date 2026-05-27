"""Tests for /api/health and /api/filters."""


def test_health_returns_200(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["physicians_loaded"] > 0
    assert data["features_loaded"] > 0


def test_health_schema(client):
    r = client.get("/api/health")
    data = r.json()
    assert set(data.keys()) == {"status", "version", "physicians_loaded", "features_loaded"}


def test_filters_returns_states_and_specialties(client):
    r = client.get("/api/filters")
    assert r.status_code == 200
    data = r.json()
    assert len(data["states"]) > 0
    assert len(data["specialties"]) > 0
    assert "Endocrinology" in data["specialties"]


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "FirstWave API" in r.json()["name"]
