"""Tests for /api/predict — the core ranking endpoint."""


def test_predict_returns_200(client):
    r = client.post("/api/predict", json={"limit": 10})
    assert r.status_code == 200


def test_predict_returns_ranked_list(client):
    r = client.post("/api/predict", json={"limit": 20})
    physicians = r.json()["physicians"]
    assert len(physicians) == 20
    scores = [p["adoption_score"] for p in physicians]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by score descending"


def test_predict_scores_in_valid_range(client):
    r = client.post("/api/predict", json={"limit": 50})
    for p in r.json()["physicians"]:
        assert 0.0 <= p["adoption_score"] <= 1.0
        assert p["predicted_days_to_rx"] > 0
        assert p["urgency_tier"] in {"Tier 1", "Tier 2", "Tier 3", "Tier 4"}


def test_predict_state_filter(client):
    r = client.post("/api/predict", json={"state": "CA", "limit": 30})
    data = r.json()
    assert all(p["state"] == "CA" for p in data["physicians"])
    assert data["filters_applied"]["state"] == "CA"


def test_predict_specialty_filter(client):
    r = client.post("/api/predict", json={"specialty": "Endocrinology", "limit": 20})
    for p in r.json()["physicians"]:
        assert p["specialty"] == "Endocrinology"


def test_predict_min_score_filter(client):
    r = client.post("/api/predict", json={"min_score": 0.5, "limit": 100})
    for p in r.json()["physicians"]:
        assert p["adoption_score"] >= 0.5


def test_predict_pagination(client):
    r1 = client.post("/api/predict", json={"limit": 5, "offset": 0})
    r2 = client.post("/api/predict", json={"limit": 5, "offset": 5})
    npis_1 = {p["NPI"] for p in r1.json()["physicians"]}
    npis_2 = {p["NPI"] for p in r2.json()["physicians"]}
    assert npis_1.isdisjoint(npis_2), "Paginated results must not overlap"


def test_predict_endocrinologists_score_higher_than_cardiologists(client):
    """Sanity check: model has learned medical context — endos > cards for GLP-1."""
    endo = client.post("/api/predict", json={"specialty": "Endocrinology", "limit": 50}).json()
    card = client.post("/api/predict", json={"specialty": "Cardiology", "limit": 50}).json()
    endo_mean = sum(p["adoption_score"] for p in endo["physicians"]) / len(endo["physicians"])
    card_mean = sum(p["adoption_score"] for p in card["physicians"]) / len(card["physicians"])
    assert endo_mean > card_mean, f"Endo {endo_mean:.3f} should beat Card {card_mean:.3f}"
