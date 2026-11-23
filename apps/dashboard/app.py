from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from apps.api.app import AppState, create_api_app


def create_dashboard_app(state: AppState | None = None) -> FastAPI:
    app = create_api_app(state)

    template_path = Path(__file__).parent / "templates" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        return "<html><body><h1>Hermes Quant Terminal</h1><p>Template missing.</p></body></html>"

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return index()

    return app


app = create_dashboard_app()
