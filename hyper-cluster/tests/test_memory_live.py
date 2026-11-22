"""Live tests for the persistent vector memory and the cloud sync."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory.sync import DriveClient, HubClient, MemorySync, SyncError
from memory.vector_memory import (
    SOURCE_DOCUMENT,
    SOURCE_FACT,
    VectorMemory,
)
from tests.cloud_doubles import CloudDouble


@pytest.fixture
def memory(tmp_path: Path) -> VectorMemory:
    return VectorMemory(tmp_path / "memory")


@pytest.fixture
def cloud() -> CloudDouble:
    double = CloudDouble()
    try:
        yield double
    finally:
        double.close()


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def test_document_is_indexed_and_found(memory: VectorMemory) -> None:
    memory.add_document(
        "architecture.md",
        "The head node is a statefulset. Workers join through the Tailscale mesh.",
    )
    hits = memory.search("how do workers join the mesh", k=3)
    assert hits
    assert "Tailscale" in hits[0].chunk.text
    assert hits[0].chunk.source == SOURCE_DOCUMENT
    assert hits[0].score > 0.0


def test_keyword_search_finds_exact_tokens(memory: VectorMemory) -> None:
    memory.add_document("models.md", "We serve Qwen2.5-32B-Instruct-GGUF on the big nodes.")
    hits = memory.search("Qwen2.5-32B-Instruct-GGUF", k=3)
    assert hits
    assert "Qwen2.5-32B" in hits[0].chunk.text
    assert hits[0].keyword > 0.0


def test_a_long_document_is_split_into_chunks(memory: VectorMemory) -> None:
    body = "\n\n".join(f"Paragraph {index}: the cluster keeps its state in the ledger." for index in range(40))
    record = memory.add_document("long.md", body)
    assert record.chunks > 1
    assert memory.stats()["chunks"] == record.chunks


def test_uploaded_files_are_indexed_automatically(memory: VectorMemory, tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("The retina display costs 480 euros.", encoding="utf-8")
    record = memory.ingest_file(path)
    assert record is not None
    assert record.name == "notes.txt"
    hits = memory.search("retina display price", k=2)
    assert hits and "480 euros" in hits[0].chunk.text


def test_unreadable_files_are_skipped(memory: VectorMemory, tmp_path: Path) -> None:
    assert memory.ingest_file(tmp_path / "missing.txt") is None


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
def test_facts_are_distilled_from_the_conversation(memory: VectorMemory) -> None:
    stored = memory.distill_facts(
        [
            {"role": "user", "content": "Remember that my name is Ada."},
            {"role": "assistant", "content": "Nice to meet you Ada."},
            {"role": "user", "content": "We use kubernetes with three node pools."},
            {"role": "user", "content": "What is the weather?"},
        ]
    )
    assert len(stored) == 2
    assert any("Ada" in fact for fact in stored)
    assert any("kubernetes" in fact for fact in stored)
    # The assistant's own words are not facts about the user.
    assert not any("Nice to meet" in fact for fact in stored)
    assert memory.stats()["by_source"][SOURCE_FACT] == 2


def test_a_fact_is_stored_only_once(memory: VectorMemory) -> None:
    stored = memory.distill_facts([{"role": "user", "content": "Remember that my name is Ada."}])
    assert stored == ["name: Ada"]
    # the same statement is never indexed twice
    again = memory.distill_facts([{"role": "user", "content": "Remember that my name is Ada."}])
    assert again == []
    assert memory.stats()["by_source"][SOURCE_FACT] == 1


def test_a_fact_outranks_a_raw_conversation(memory: VectorMemory) -> None:
    memory.record_exchange("what is my setup", "you run a cluster")
    memory.add_fact("setup: a three node kubernetes cluster")
    hits = memory.search("setup", k=2)
    assert hits
    assert hits[0].chunk.source == SOURCE_FACT


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_the_memory_survives_a_restart(tmp_path: Path) -> None:
    first = VectorMemory(tmp_path / "memory")
    first.add_document("a.md", "The cluster is called hyper-cluster-omega.")
    first.record_exchange("q", "a")
    second = VectorMemory(tmp_path / "memory")
    assert second.stats()["chunks"] == first.stats()["chunks"]
    assert second.revision == first.revision
    hits = second.search("hyper-cluster-omega", k=2)
    assert hits


def test_a_snapshot_can_be_restored_into_another_memory(tmp_path: Path) -> None:
    source = VectorMemory(tmp_path / "source")
    source.add_document("a.md", "The origin of the cluster is a single GPU.")
    snapshot = source.snapshot()
    target = VectorMemory(tmp_path / "target")
    assert target.restore(snapshot) == 1
    assert target.search("single GPU", k=1)


def test_the_context_block_names_the_source(memory: VectorMemory) -> None:
    memory.add_document("pricing.md", "The pro plan costs 20 euros per month.")
    context = memory.context("how much is the pro plan", k=2)
    assert "MEMORY CONTEXT" in context
    assert "pricing.md" in context
    assert "20 euros" in context


# --------------------------------------------------------------------------- #
# The cloud sync
# --------------------------------------------------------------------------- #
def _sync(cloud: CloudDouble) -> MemorySync:
    return MemorySync(
        drive=DriveClient(
            token="drive-token",
            api_base=cloud.drive_api_base,
            upload_base=cloud.drive_upload_base,
        ),
        hub=HubClient(token="hub-token", base_url=cloud.hub_base_url),
        drive_folder="Hyper-Cluster Memory",
        hub_repo="someone/hyper-cluster",
    )


async def test_the_snapshot_reaches_drive_and_the_hub(memory: VectorMemory, cloud: CloudDouble) -> None:
    memory.add_document("a.md", "The cluster syncs itself to the cloud.")
    receipts = await _sync(cloud).push(memory.snapshot())
    assert {receipt.target for receipt in receipts} == {"drive", "hub"}
    assert all(receipt.ok for receipt in receipts)
    drive_files = [item for item in cloud.files() if item["name"].endswith(".json")]
    assert drive_files
    assert drive_files[0]["appProperties"]["kind"] == "memory-snapshot"
    assert drive_files[0]["size"] > 0
    hub_files = [item for item in cloud.hub_files() if item["path"].endswith(".json")]
    assert hub_files
    assert len(hub_files[0]["content"]) > 0


async def test_a_pulled_snapshot_restores_the_memory(memory: VectorMemory, cloud: CloudDouble) -> None:
    memory.add_document("a.md", "The cluster syncs itself to the cloud.")
    sync = _sync(cloud)
    await sync.push(memory.snapshot())

    target = VectorMemory(cloud.base_url and memory.path.parent / "restored")
    snapshots = await sync.pull()
    assert snapshots
    assert target.restore(snapshots[0]) == memory.stats()["chunks"]
    assert target.search("syncs itself", k=1)


async def test_a_corrupted_download_is_rejected(memory: VectorMemory, cloud: CloudDouble) -> None:
    memory.add_document("a.md", "The cluster syncs itself to the cloud.")
    sync = _sync(cloud)
    await sync.push(memory.snapshot())
    snapshot_file = next(item for item in cloud.files() if item["name"].endswith(".json"))
    cloud.corrupt(snapshot_file["id"])
    snapshots = await sync.pull(targets=("drive",))
    assert snapshots == []


async def test_the_checksum_of_the_snapshot_is_verified(memory: VectorMemory, cloud: CloudDouble) -> None:
    sync = _sync(cloud)
    snapshot = memory.snapshot()
    snapshot["chunks"] = [{"id": "x", "text": "tampered"}]
    with pytest.raises(SyncError):
        sync._verify(json.dumps(snapshot).encode("utf-8"))


async def test_pulling_without_a_snapshot_is_not_an_error(memory: VectorMemory, cloud: CloudDouble) -> None:
    snapshots = await _sync(cloud).pull()
    assert snapshots == []
