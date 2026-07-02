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


def test_embed_post_shape():
    # POST /embed is the path corpus docs use — a long plot as a GET query param would risk a 414.
    r = client.post("/embed", json={"text": "a bank robbery"})
    assert r.status_code == 200
    body = r.json()
    assert body["dims"] == DIMS
    assert body["model"] == "bge-m3"
    assert len(body["vector"]) == DIMS
    assert all(-127 <= x <= 127 for x in body["vector"])


def test_embed_post_matches_get():
    # Same text → same vector whether sent via GET query param or POST body.
    text = "a heist crew plans one last job"
    assert client.post("/embed", json={"text": text}).json()["vector"] == \
        client.get("/embed", params={"text": text}).json()["vector"]


def test_embed_post_long_plot_no_uri_limit():
    # A multi-thousand-char plot must embed fine via POST (the whole point — no 414).
    long_plot = "The team descends through layered dreams. " * 300  # ~12k chars
    r = client.post("/embed", json={"text": long_plot})
    assert r.status_code == 200
    assert len(r.json()["vector"]) == DIMS


def test_embed_batch_shape():
    r = client.post("/embed/batch", json={"texts": ["a bank robbery", "", "a romance"]})
    assert r.status_code == 200
    body = r.json()
    assert body["dims"] == DIMS
    assert body["model"] == "bge-m3"
    assert len(body["vectors"]) == 3
    assert all(len(v) == DIMS for v in body["vectors"])
    assert body["vectors"][1] == [0] * DIMS
