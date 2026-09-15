from fastapi.testclient import TestClient

from research_rag.api import app


def test_frontend_and_static_assets_are_served() -> None:
    client = TestClient(app)

    page = client.get("/")
    styles = client.get("/static/styles.css")
    script = client.get("/static/app.js")

    assert page.status_code == 200
    assert "ResearchGraph" in page.text
    assert 'data-testid="question-input"' in page.text
    assert "mathjax@3.2.2" in page.text
    assert "Retrieval pipeline" not in page.text
    assert "Library online" not in page.text
    assert styles.status_code == 200
    assert "--coral" in styles.text
    assert "mjx-container" in styles.text
    assert script.status_code == 200
    assert 'fetch("/hybrid/ask"' in script.text
    assert 'fetch("/health"' not in script.text
    assert "typesetMath" in script.text
