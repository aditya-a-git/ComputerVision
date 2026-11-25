# Implementation record
Baseline 3122e28ba48d81ba4feb60944e6f3aa2d327ae1c (see history for full stage hashes).
Stage 0 introduces a credential-free homepage. Smoke test: python -m pytest -q. Data, model and CARD access follow in subsequent stages.
Stage 0 9fbc573: server health returned ok. Initial AppTest resolved app.py relative to tests and failed. Corrected its absolute path in the following fix commit.
