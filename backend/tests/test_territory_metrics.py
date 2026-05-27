"""Tests for /api/territory and /api/metrics."""


def test_territory_returns_states(client):
    r = client.get("/api/territory")
    assert r.status_code == 200
    territories = r.json()["territories"]
    assert len(territories) > 0


def test_territory_sorted_by_score(client):
    r = client.get("/api/territory")
    scores = [t["mean_adoption_score"] for t in r.json()["territories"]]
    assert scores == sorted(scores, reverse=True)


def test_territory_aggregates_valid(client):
    r = client.get("/api/territory")
    for t in r.json()["territories"]:
        assert t["physician_count"] > 0
        assert 0.0 <= t["mean_adoption_score"] <= 1.0
        assert t["tier_1_count"] >= 0
        assert t["high_potential_count"] >= 0


def test_metrics_returns_model_performance(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["xgboost_auc"] > 0.5, "Model must outperform random"
    assert m["xgboost_pr_auc"] > 0.0
    assert m["cox_concordance"] > 0.5
    assert m["test_size"] > 0


def test_metrics_lift_curve_decreasing(client):
    """Top deciles should have higher lift than bottom deciles."""
    r = client.get("/api/metrics")
    lift = r.json()["lift_by_decile"]
    decile_1 = lift.get("1", 0)
    decile_10 = lift.get("10", 0)
    assert decile_1 > decile_10, "Top decile lift must exceed bottom decile"


def test_metrics_includes_feature_importance(client):
    r = client.get("/api/metrics")
    importance = r.json()["feature_importance"]
    assert len(importance) > 0
    top_feature = max(importance, key=importance.get)
    assert "Endocrinology" in top_feature or "glp1" in top_feature.lower(), (
        f"Top feature should be medically meaningful, got: {top_feature}"
    )
