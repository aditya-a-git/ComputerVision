# Implementation record
Baseline 3122e28ba48d81ba4feb60944e6f3aa2d327ae1c (see history for full stage hashes).
Stage 0 introduces a credential-free homepage. Smoke test: python -m pytest -q. Data, model and CARD access follow in subsequent stages.
Stage 0 9fbc573: server health returned ok. Initial AppTest resolved app.py relative to tests and failed. Corrected its absolute path in the following fix commit.
Stage 1: inspected all 38,383 images, HF train/validation partitions, five classes and 61,075 boxes. No expected count deviations. Dry-run selected 8 images. CLI uses serial requests, an aggregate byte budget, validated exact-name local imports and a signed-URL-free manifest. Run python -m scripts.inspect_rdd and python -m scripts.prepare_rdd --dry-run.
Stage 1 commit 58510bb omitted core/data because of an overly broad ignore pattern. The immediate follow-up tracks the ingestion module and scopes the generated-data exclusion to the repository root.
