from fastapi.testclient import TestClient

from recommender_drift_monitor.api import create_app
from recommender_drift_monitor.service import DemoService


def make_client() -> TestClient:
    return TestClient(create_app(DemoService()))


def test_health_endpoint() -> None:
    response = make_client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_overview_and_drift_report() -> None:
    client = make_client()
    overview = client.get("/overview")
    report = client.get("/drift/report")
    assert overview.status_code == 200
    assert overview.json()["item_count"] == 42
    assert report.status_code == 200
    assert len(report.json()["features"]) == 2


def test_simulate_and_reset_drift() -> None:
    client = make_client()
    simulated = client.post("/drift/simulate", json={"shifted": True})
    assert simulated.status_code == 200
    assert simulated.json()["drift_detected"] is True

    reset = client.post("/drift/reset")
    assert reset.status_code == 200
    assert reset.json()["status"] == "STABLE"


def test_recommend_endpoint() -> None:
    response = make_client().get("/recommend/7?limit=4")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 7
    assert body["count"] == 4
    assert len(body["recommendations"]) == 4


def test_recommend_limit_validation() -> None:
    assert make_client().get("/recommend/7?limit=50").status_code == 422
