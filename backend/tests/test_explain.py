"""Tests for /api/explain/{npi} and /api/physicians/{npi}."""


def test_explain_returns_shap_values(client, sample_npi):
    r = client.get(f"/api/explain/{sample_npi}")
    assert r.status_code == 200
    data = r.json()
    assert data["npi"] == sample_npi
    assert "expected_value" in data
    assert len(data["contributions"]) > 0


def test_explain_contributions_sorted_by_abs_shap(client, sample_npi):
    r = client.get(f"/api/explain/{sample_npi}")
    abs_vals = [c["abs_shap"] for c in r.json()["contributions"]]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_explain_each_contribution_schema(client, sample_npi):
    r = client.get(f"/api/explain/{sample_npi}")
    for c in r.json()["contributions"]:
        assert "feature" in c
        assert "value" in c
        assert "shap_value" in c
        assert "abs_shap" in c
        assert c["abs_shap"] >= 0


def test_explain_unknown_npi_returns_404(client):
    r = client.get("/api/explain/9999999999")
    assert r.status_code == 404


def test_physician_detail_returns_full_record(client, sample_npi):
    r = client.get(f"/api/physicians/{sample_npi}")
    assert r.status_code == 200
    data = r.json()
    assert data["NPI"] == sample_npi
    assert data["gender"] in ("M", "F")
    assert data["years_in_practice"] > 0


def test_physician_detail_unknown_npi(client):
    r = client.get("/api/physicians/9999999999")
    assert r.status_code == 404
