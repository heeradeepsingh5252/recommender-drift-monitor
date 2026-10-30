"""FastAPI routes for recommendations and drift-monitor controls."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from recommender_drift_monitor.service import DemoService


class DriftSimulationRequest(BaseModel):
    shifted: bool = Field(
        default=True,
        description="True simulates a changed genre and rating distribution; false resets the baseline.",
    )


def create_app(service: DemoService | None = None) -> FastAPI:
    """Create an API app. Injection makes tests isolated and repeatable."""
    demo = service or DemoService()
    api = FastAPI(
        title="Recommender Drift Monitor API",
        description=(
            "A portfolio demo for item-item recommendations and synthetic data drift monitoring. "
            "All data is generated locally for demonstration."
        ),
        version="1.0.0",
    )

    @api.get("/health", tags=["system"])
    def health() -> dict[str, object]:
        return demo.health()

    @api.get("/overview", tags=["monitoring"])
    def overview() -> dict[str, object]:
        return demo.overview()

    @api.get("/drift/report", tags=["monitoring"])
    def drift_report() -> dict[str, object]:
        return demo.drift_report()

    @api.post("/drift/simulate", tags=["monitoring"])
    def simulate_drift(request: DriftSimulationRequest) -> dict[str, object]:
        return demo.simulate_drift(shifted=request.shifted)

    @api.post("/drift/reset", tags=["monitoring"])
    def reset_drift() -> dict[str, object]:
        return demo.simulate_drift(shifted=False)

    @api.get("/recommend/{user_id}", tags=["recommendations"])
    def recommend(
        user_id: int,
        limit: int = Query(default=10, ge=1, le=25, description="Number of results to return."),
    ) -> dict[str, object]:
        try:
            results = demo.recommendations(user_id=user_id, limit=limit)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"user_id": user_id, "count": len(results), "recommendations": results}

    return api


app = create_app()
