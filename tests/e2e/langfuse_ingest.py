"""Latest relay Langfuse pull -> a running metergraph server, read back.

Drives the real ``pull_langfuse`` loop against the recorded Langfuse fixture
(tests/fixtures/langfuse_v2, served as one stubbed page, so no Langfuse
credential is needed), pushes the JSONL to ``MG_URL`` and reads every row back
through ``/v1/calls``. What survives that round trip is the actual relay to
server contract; unit tests only prove what the relay writes to disk.

What this is here to catch:

* **Tags.** The server keeps only string values in ``tags``. Langfuse tags used
  to travel as one list and were dropped whole; each must now land as its own
  ``langfuse.tag.<tag>: "true"`` pair, next to the customer's metadata
  (``langfuse.metadata.<key>``).
* **Timing.** ``latency_ms`` and ``ttft_ms`` come from Langfuse's seconds.
* **Every cost-bearing observation lands**, embeddings included, with the
  provider resolved and Langfuse's cost carried as ``reported_cost_usd``.
* **Attribution**, where the server's read returns it: ``user_id``.
* **Re-syncs do not duplicate.** The same pull is pushed twice; every
  observation must be stored once, deduplicated on its import identity.

Run it against a server whose ``/v1/calls`` returns ``tags``, ``ttft_ms``,
``user_id`` and ``reported_cost_usd``: metergraph's hosted app does; the OSS
server's read does not return them yet, so this is not part of the OSS CI job.

Set ``MG_URL`` and ``MG_TOKEN`` (a key with ingest and read scope). ``MG_REPORT`` names a file to write the
read-back rows to (cost status and catalog price per row), for comparing a
server's pricing with Langfuse's.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from metergraphrelay.providers import langfuse
from metergraphrelay.providers.langfuse import COST_BEARING_TYPES, pull_langfuse
from metergraphrelay.push import push_file

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "langfuse_v2"
ROUTE_PREFIX = "relay.langfuse-e2e"


def _fixture_page(base_url, *, public_key, secret_key, params, sleep=None):
    conditions = json.loads(params["filter"])
    assert conditions[0]["value"] == list(COST_BEARING_TYPES), conditions
    observations = json.loads((FIXTURE_DIR / "observations.json").read_text())
    return {"data": observations["data"], "meta": {"cursor": None}}


def _read_rows(base_url: str, token: str, route: str) -> list[dict]:
    query = urllib.parse.urlencode({"route": route, "limit": 500})
    request = urllib.request.Request(
        f"{base_url}/v1/calls?{query}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)
    return [item for item in payload["items"] if item["route"] == route]


def _read_back(base_url: str, token: str, route: str) -> dict[str, dict]:
    return {item["request_id"]: item for item in _read_rows(base_url, token, route)}


def main() -> None:
    base_url = os.environ["MG_URL"].rstrip("/")
    token = os.environ["MG_TOKEN"]
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    cost_keys = [k for k, v in manifest["observations"].items() if v.get("cost_bearing")]
    route = f"{ROUTE_PREFIX}.{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "langfuse.jsonl"
        with patch.object(langfuse, "fetch_observations_page", _fixture_page):
            imported, skipped = pull_langfuse(
                base_url="https://cloud.langfuse.com",
                public_key="not-used-by-the-stub",
                secret_key="not-used-by-the-stub",
                count=1000,
                since=None,
                until="2026-01-02T00:00:00Z",
                trace_names=[],
                tags=[],
                environment=None,
                route=route,
                output_path=str(output),
            )
        assert (imported, skipped) == (len(cost_keys), 0), (imported, skipped)
        pushed, failed = push_file(str(output), token, base_url=base_url)
        assert (pushed, failed) == (imported, 0), (pushed, failed)
        # A re-sync of the same window: the server must drop every row.
        pushed, failed = push_file(str(output), token, base_url=base_url)
        assert (pushed, failed) == (imported, 0), (pushed, failed)

    # A server with a queue (the hosted app's worker) stores rows a little
    # after the push returns; wait for all of them, within a bound.
    expected_ids = {manifest["observations"][key]["id"] for key in cost_keys}
    deadline = time.monotonic() + float(os.environ.get("MG_WAIT_SECONDS", "90"))
    stored = _read_back(base_url, token, route)
    while not expected_ids <= stored.keys() and time.monotonic() < deadline:
        time.sleep(2)
        stored = _read_back(base_url, token, route)
    # Give the second push's batch time to be processed (and dropped) too.
    time.sleep(float(os.environ.get("MG_DEDUP_SETTLE_SECONDS", "10")))
    rows = _read_rows(base_url, token, route)
    duplicated = sorted(
        request_id for request_id in {r["request_id"] for r in rows}
        if sum(r["request_id"] == request_id for r in rows) > 1
    )
    assert not duplicated, f"re-sync stored these observations twice: {duplicated}"
    stored = {r["request_id"]: r for r in rows}
    report = []
    for key in cost_keys:
        spec = manifest["observations"][key]
        readback = manifest["langfuse_readback"][key]
        trace = manifest["traces"][spec["trace"]]
        item = stored.get(spec["id"])
        assert item is not None, f"{key} did not land"
        tags = item["tags"] or {}
        for tag in trace["tags"]:
            assert tags.get(f"langfuse.tag.{tag}") == "true", (key, tags)
        if spec["trace"] == "main":
            assert tags.get("langfuse.metadata.customer_tier") == "gold", (key, tags)
        if "user_id" in item and trace.get("user_id"):
            assert item["user_id"] == trace["user_id"], (key, item["user_id"])
        if readback.get("latency") is not None:
            assert item["latency_ms"] == round(readback["latency"] * 1000), (key, item)
        if readback.get("timeToFirstToken") is not None:
            assert item["ttft_ms"] == round(readback["timeToFirstToken"] * 1000), (key, item)
        assert item["provider"] in spec["expected_providers"], (key, item["provider"])
        reported = item.get("reported_cost_usd")
        assert reported is not None, f"{key}: Langfuse's cost was not kept"
        assert Decimal(str(reported)) == Decimal(str(readback["totalCost"])), (key, reported)
        report.append({
            "key": key,
            "provider": item["provider"],
            "model": item["model"],
            "langfuse_cost": readback["totalCost"],
            "cost_usd": item.get("cost_usd"),
            "reported_cost_usd": reported,
            "catalog_cost_usd": item.get("catalog_cost_usd"),
            "cost_status": item.get("cost_status"),
            "catalog_reasons": item.get("catalog_reasons"),
            "latency_ms": item["latency_ms"],
            "ttft_ms": item["ttft_ms"],
            "user_id": item.get("user_id"),
            "tag_pairs": len(tags),
        })
    if os.environ.get("MG_REPORT"):
        Path(os.environ["MG_REPORT"]).write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"langfuse e2e: {len(report)} rows round-tripped with tags, timing and cost;"
        " a second push of the same rows stored none"
    )


if __name__ == "__main__":
    main()
