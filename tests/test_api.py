"""Contract-shape tests via FastAPI's TestClient. Other repos depend on these
exact shapes, so we assert keys, dims, and int8 bounds rather than values.
"""

from fastapi.testclient import TestClient

from server import DIMS, app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "bge-m3", "dims": DIMS}


def test_embed_shape():
    r = client.get("/embed", params={"text": "a bank robbery"})
    assert r.status_code == 200
    body = r.json()
    assert body["dims"] == DIMS
    assert body["model"] == "bge-m3"
    assert len(body["vector"]) == DIMS
    assert all(-127 <= x <= 127 for x in body["vector"])


def test_embed_empty_is_zero_vector():
    r = client.get("/embed", params={"text": ""})
    assert r.json()["vector"] == [0] * DIMS


def test_embed_batch_shape():
    r = client.post("/embed/batch", json={"texts": ["a bank robbery", "", "a romance"]})
    assert r.status_code == 200
    body = r.json()
    assert body["dims"] == DIMS
    assert body["model"] == "bge-m3"
    assert len(body["vectors"]) == 3
    assert all(len(v) == DIMS for v in body["vectors"])
    assert body["vectors"][1] == [0] * DIMS
