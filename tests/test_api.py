"""Tests del servicio FastAPI: contratos, modo simplificado, historia y errores en español."""
import copy

from fastapi.testclient import TestClient

from api.examples import HISTORY_EXAMPLE
from api.main import app


def test_root_redirects_to_app():
    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert response.headers["location"] == "/app/"


def test_health_reports_ready():
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["n_features"] > 100


def test_features_catalog_matches_health():
    with TestClient(app) as client:
        health = client.get("/health").json()
        catalog = client.get("/features").json()
        assert catalog["n_features"] == health["n_features"]
        assert len(catalog["features"]) == catalog["n_features"]


def test_predict_fills_missing_features_and_returns_plausible_value():
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": {}})
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["predicted_silica"] <= 6.0
        assert body["n_features_provided"] == 0
        assert body["n_features_filled"] > 100


def test_predict_rejects_unknown_feature_in_spanish():
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": {"not_a_real_feature": 1.0}})
        assert response.status_code == 422
        assert "desconocidas" in response.json()["detail"]


def test_predict_from_history_with_real_example():
    with TestClient(app) as client:
        response = client.post("/predict-from-history", json=HISTORY_EXAMPLE)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["predicted_silica"] <= 10.0
        assert body["n_hours_received"] == len(HISTORY_EXAMPLE["history"])
        assert body["n_features_computed"] > 0


def test_predict_from_history_requires_lab_results():
    # Sin el laboratorio de una hora intermedia, el error debe nombrar el campo y la hora.
    payload = copy.deepcopy(HISTORY_EXAMPLE)
    del payload["history"][1]["silica"]
    with TestClient(app) as client:
        response = client.post("/predict-from-history", json=payload)
        assert response.status_code == 422
        assert "silica" in response.json()["detail"]


def test_validation_errors_are_in_spanish():
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": "no-es-un-objeto"})
        assert response.status_code == 422
        assert "solicitud" in response.json()["detail"]


def test_simulate_returns_delta():
    with TestClient(app) as client:
        response = client.post(
            "/simulate",
            json={"base_features": {}, "deltas": {"Amina Flow__mean": 100.0}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "delta_silica" in body
        assert body["applied"]["Amina Flow__mean"] is not None


def test_reference_endpoint_exposes_bands_and_residuals():
    with TestClient(app) as client:
        body = client.get("/reference").json()
        for key in ("reference", "min", "max", "p5", "p95"):
            assert key in body and body[key]
        assert body["mae_test"] > 0
        # Muestra de residuos de test para estimar empíricamente la probabilidad de umbral.
        assert isinstance(body["residuals_test"], list) and len(body["residuals_test"]) > 0


def test_explain_returns_signed_sorted_contributions():
    with TestClient(app) as client:
        response = client.post("/explain", json={"features": {}})
        assert response.status_code == 200
        body = response.json()
        assert 0 < len(body["contributions"]) <= 8
        assert {"feature", "label", "value", "shap_value"} <= set(body["contributions"][0])
        magnitudes = [abs(c["shap_value"]) for c in body["contributions"]]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert isinstance(body["base_value"], float)


def test_explain_labels_are_well_formed():
    """Cada etiqueta legible debe tener los paréntesis cerrados y nada de guiones largos."""
    with TestClient(app) as client:
        body = client.post(
            "/explain",
            json={"features": {"% Silica Concentrate__lag1h": 2.0, "Amina Flow__mean": 500}},
        ).json()
        for contribution in body["contributions"]:
            label = contribution["label"]
            assert label.count("(") == label.count(")"), f"paréntesis desbalanceado: {label}"
            assert "—" not in label


def test_report_returns_structured_sections():
    with TestClient(app) as client:
        body = client.post("/report", json={"features": {}}).json()
        assert isinstance(body["is_fallback"], bool)
        report = body["report"]
        assert "situation" in report
        # Con entrada válida (completada con la referencia) el grafo produce secciones.
        assert "drivers" in report and "recommendation" in report


def test_simulate_sustained_returns_8h_trajectories():
    with TestClient(app) as client:
        body = client.post(
            "/simulate-sustained",
            json={"base_features": {}, "deltas": {"Ore Pulp Density__mean": 0.1}},
        ).json()
        assert len(body["trajectory"]) == 8
        assert len(body["trajectory_base"]) == 8
        assert "delta_accumulated" in body
