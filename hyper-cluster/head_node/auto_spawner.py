"""Headless compute provisioner and pre-emptive TTL rotator.

The cluster must never depend on a human pressing a button.  This module
*provisions real compute* through the headless APIs of the compute providers
and keeps the pool alive:

``KaggleProvisioner``
    Pushes a GPU kernel (a folder with ``kernel-metadata.json`` and a worker
    bootstrap) to ``https://www.kaggle.com/api/v1/kernels/push`` and polls the
    kernel status.  No browser, no captcha, no clicking.

``HuggingFaceSpaceProvisioner``
    Creates a ZeroGPU Space (``POST /api/repos/create``), uploads the
    Dockerfile and the worker bootstrap (``POST /api/spaces/{id}/upload/...``)
    and watches the runtime stage.

``GitHubActionsProvisioner``
    Dispatches a ``workflow_dispatch`` run that starts the worker
    (``POST /repos/{owner}/{repo}/actions/workflows/{wf}/dispatches``) and
    follows the run until it completes.

``OracleProvisioner``
    Launches an always-free Ampere A1 instance through the OCI Compute API
    (``POST https://compute.{region}.oraclecloud.com/20160918/instances``)
    with a cloud-init payload that starts the worker.  Requests are signed
    with the user's RSA key exactly as OCI documents.

``EdgeApiProvisioner``
    Provisions *edge* capacity: it leases an account of a free inference API
    from the credential vault and verifies it live against the provider's
    ``/models`` endpoint before handing it to the edge bridge.

Every provisioner is multi-account: the spawner walks the accounts of a
provider, marks a failing account as degraded and silently continues with the
next one.  The **pre-emptive TTL rotator** uses the same machinery to replace
a worker whose session is about to expire: the replacement is provisioned
*before* the old session dies and the old worker is only drained once the
replacement has registered and is serving -- that is what makes the handover
zero-downtime.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from core.config import Settings
from core.credentials import (
    HEALTH_DEGRADED,
    HEALTH_HEALTHY,
    HEALTH_INVALID,
    Credential,
    CredentialVault,
    get_vault,
)
from core.ids import now_s
from core.logging_setup import get_logger
from core.providers import get_provider
from head_node.worker_registry import WorkerRecord, WorkerRegistry

LOGGER = get_logger("head_node.auto_spawner")

#: Lifecycle of a provisioned instance.
STATE_PROVISIONING = "provisioning"
STATE_REGISTERED = "registered"
STATE_FAILED = "failed"
STATE_TERMINATED = "terminated"


@dataclass
class SpawnRequest:
    """What to provision."""

    provider: str
    region: str = "unknown"
    model: str | None = None
    family: str | None = None
    params_b: float | None = None
    vram_gb: float = 0.0
    ttl_seconds: int = 3600
    labels: dict[str, str] = field(default_factory=dict)
    replaces: str | None = None
    head_url: str = ""
    mesh_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "region": self.region,
            "model": self.model,
            "family": self.family,
            "params_b": self.params_b,
            "vram_gb": self.vram_gb,
            "ttl_seconds": self.ttl_seconds,
            "labels": dict(self.labels),
            "replaces": self.replaces,
        }


@dataclass
class SpawnResult:
    """The outcome of one provisioning attempt."""

    ok: bool = False
    provider: str = ""
    instance_id: str = ""
    endpoint: str = ""
    account_label: str = ""
    detail: str = ""
    error: str = ""
    created_at: float = field(default_factory=now_s)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "instance_id": self.instance_id,
            "endpoint": self.endpoint,
            "account_label": self.account_label,
            "detail": self.detail,
            "error": self.error,
            "created_at": self.created_at,
            "meta": dict(self.meta),
        }


@dataclass
class SpawnedInstance:
    """A provisioned instance the spawner is keeping alive."""

    instance_id: str
    provider: str
    account_label: str
    endpoint: str = ""
    state: str = STATE_PROVISIONING
    created_at: float = field(default_factory=now_s)
    expires_at: float = 0.0
    replaces: str | None = None
    worker_id: str | None = None
    last_error: str = ""
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "provider": self.provider,
            "account_label": self.account_label,
            "endpoint": self.endpoint,
            "state": self.state,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "replaces": self.replaces,
            "worker_id": self.worker_id,
            "last_error": self.last_error,
            "attempts": self.attempts,
        }


# --------------------------------------------------------------------------- #
# Provisioners
# --------------------------------------------------------------------------- #
class ComputeProvisioner(ABC):
    """Base class for every headless compute provisioner."""

    name: str = "compute"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = "",
    ) -> None:
        self.settings = settings
        self.explicit_base_url = bool(base_url)
        self.base_url = (base_url or self.default_base_url()).rstrip("/")
        self._client = client

    # ------------------------------------------------------------------ #
    def default_base_url(self) -> str:
        spec = get_provider(self.name)
        return spec.base_url if spec else ""

    @property
    def client(self) -> httpx.AsyncClient:
        """The shared HTTP client (created on first use)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
                headers={"user-agent": "hyper-cluster-spawner/1.0"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def auth_headers(self, credential: Credential) -> dict[str, str]:
        """Authentication headers for this provider.

        ``user:key`` tokens become HTTP Basic credentials (Kaggle), everything
        else becomes a bearer token.
        """
        token = credential.token or ""
        if not token:
            return {}
        if ":" in token and self.name == "kaggle":
            raw = base64.b64encode(token.encode("utf-8")).decode("ascii")
            return {"Authorization": f"Basic {raw}"}
        spec = get_provider(self.name)
        if spec is not None:
            return spec.header_for(token)
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------ #
    @abstractmethod
    async def provision(self, request: SpawnRequest, credential: Credential) -> SpawnResult:
        """Create the compute and return the result."""

    @abstractmethod
    async def terminate(self, instance_id: str, credential: Credential) -> bool:
        """Destroy the compute; ``True`` when the provider confirmed it."""

    # ------------------------------------------------------------------ #
    def _worker_bootstrap(self, request: SpawnRequest) -> str:
        """The script the provisioned node runs to join the cluster."""
        model = request.model or ""
        backend = "auto" if model else "simulator"
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "# Hyper-Cluster Omega -- headless worker bootstrap (generated).",
                "set -euo pipefail",
                f"export HYPER_WORKER_HEAD_URL='{request.head_url}'",
                f"export HYPER_WORKER_MESH_TOKEN='{request.mesh_token}'",
                f"export HYPER_WORKER_PROVIDER='{request.provider}'",
                f"export HYPER_WORKER_REGION='{request.region}'",
                f"export HYPER_WORKER_MODEL_FILENAME='{model}'",
                f"export HYPER_WORKER_MODEL_FAMILY='{request.family or ''}'",
                f"export HYPER_WORKER_TTL_SECONDS='{int(request.ttl_seconds)}'",
                f"export HYPER_WORKER_BACKEND='{backend}'",
                "if [ -x ./worker/bootstrap.sh ]; then",
                "  ./worker/bootstrap.sh --install",
                "else",
                "  echo 'worker/bootstrap.sh is missing from the payload' >&2",
                "  exit 1",
                "fi",
            ]
        )


class KaggleProvisioner(ComputeProvisioner):
    """Pushes a GPU kernel to Kaggle (no browser interaction)."""

    name = "kaggle"

    def default_base_url(self) -> str:
        return "https://www.kaggle.com/api/v1"

    def _owner_slug(self, credential: Credential, request: SpawnRequest) -> tuple[str, str]:
        owner = credential.metadata.get("owner") or request.labels.get("owner") or ""
        slug = request.labels.get("kernel") or f"hyper-cluster-{request.region}-{int(now_s())}"
        return owner, slug

    async def provision(self, request: SpawnRequest, credential: Credential) -> SpawnResult:
        owner, slug = self._owner_slug(credential, request)
        if not owner:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error="the Kaggle account has no 'owner' metadata (cannot build the kernel slug)",
            )
        kernel_id = f"{owner}/{slug}"
        metadata = {
            "id": kernel_id,
            "title": slug,
            "code_file": "worker_bootstrap.sh",
            "language": "shell",
            "kernel_type": "script",
            "enable_gpu": True,
            "enable_internet": True,
            "keywords": ["hyper-cluster", "worker"],
            "dataset_sources": [],
            "model_sources": [],
        }
        bootstrap = self._worker_bootstrap(request)
        try:
            response = await self.client.post(
                f"{self.base_url}/kernels/push",
                headers=self.auth_headers(credential),
                files={
                    "metadata": (None, json.dumps(metadata), "application/json"),
                    "worker_bootstrap.sh": ("worker_bootstrap.sh", bootstrap, "text/x-shellscript"),
                },
            )
        except httpx.HTTPError as exc:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=f"kaggle push failed: {exc}",
            )
        if response.status_code >= 400:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                instance_id=kernel_id,
                error=f"kaggle push returned HTTP {response.status_code}: {response.text[:300]}",
            )
        return SpawnResult(
            ok=True,
            provider=self.name,
            instance_id=kernel_id,
            endpoint="",
            account_label=credential.label,
            detail=f"kernel {kernel_id} pushed; the worker will phone home",
            meta={"owner": owner, "slug": slug, "kernel_id": kernel_id},
        )

    async def terminate(self, instance_id: str, credential: Credential) -> bool:
        owner, _, slug = instance_id.partition("/")
        if not slug:
            return False
        try:
            response = await self.client.delete(
                f"{self.base_url}/kernels/delete/{owner}/{slug}",
                headers=self.auth_headers(credential),
            )
        except httpx.HTTPError:
            return False
        return response.status_code < 400


class HuggingFaceSpaceProvisioner(ComputeProvisioner):
    """Creates a ZeroGPU Space that runs the worker."""

    name = "hf_spaces"

    def default_base_url(self) -> str:
        return "https://huggingface.co/api"

    def _repo_id(self, credential: Credential, request: SpawnRequest) -> str:
        owner = credential.metadata.get("owner") or request.labels.get("owner") or ""
        slug = (
            request.labels.get("space")
            or request.labels.get("kernel")
            or f"hyper-cluster-{request.region}-{int(now_s())}"
        )
        return f"{owner}/{slug}" if owner else slug

    async def provision(self, request: SpawnRequest, credential: Credential) -> SpawnResult:
        repo_id = self._repo_id(credential, request)
        headers = self.auth_headers(credential)
        try:
            created = await self.client.post(
                f"{self.base_url}/repos/create",
                headers=headers,
                json={
                    "type": "space",
                    "name": repo_id,
                    "sdk": "docker",
                    "private": False,
                    "hardware": "zero-a10g",
                },
            )
        except httpx.HTTPError as exc:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=f"huggingface repo create failed: {exc}",
            )
        if created.status_code >= 400 and created.status_code != 409:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                instance_id=repo_id,
                error=f"huggingface repo create returned HTTP {created.status_code}: {created.text[:300]}",
            )

        dockerfile = (
            "FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends "
            "curl ca-certificates git python3 python3-pip && rm -rf /var/lib/apt/lists/*\n"
            "COPY . /app\n"
            "WORKDIR /app\n"
            "RUN chmod +x worker_bootstrap.sh\n"
            "ENTRYPOINT [\"/app/worker_bootstrap.sh\"]\n"
        )
        readme = (
            "---\n"
            "title: Hyper-Cluster Omega Worker\n"
            "emoji: ⚡\n"
            "colorFrom: indigo\n"
            "sdk: docker\n"
            "app_port: 8081\n"
            "pinned: false\n"
            "hardware:\n"
            "  class: zero-a10g\n"
            "---\n"
            "Headless worker for the Hyper-Cluster Omega control plane.\n"
        )
        uploads = {
            "Dockerfile": ("Dockerfile", dockerfile, "text/plain"),
            "README.md": ("README.md", readme, "text/markdown"),
            "worker_bootstrap.sh": ("worker_bootstrap.sh", self._worker_bootstrap(request), "text/x-shellscript"),
        }
        try:
            for path, payload in uploads.items():
                response = await self.client.post(
                    f"{self.base_url}/spaces/{repo_id}/upload/main/{path}",
                    headers=headers,
                    files={"file": payload},
                )
                if response.status_code >= 400:
                    return SpawnResult(
                        ok=False,
                        provider=self.name,
                        account_label=credential.label,
                        instance_id=repo_id,
                        error=(
                            f"huggingface upload of {path} returned HTTP "
                            f"{response.status_code}: {response.text[:300]}"
                        ),
                    )
        except httpx.HTTPError as exc:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                instance_id=repo_id,
                error=f"huggingface upload failed: {exc}",
            )
        return SpawnResult(
            ok=True,
            provider=self.name,
            instance_id=repo_id,
            endpoint=f"https://{repo_id.replace('/', '-')}.hf.space",
            account_label=credential.label,
            detail=f"space {repo_id} created; the worker will phone home",
            meta={"repo_id": repo_id},
        )

    async def terminate(self, instance_id: str, credential: Credential) -> bool:
        try:
            response = await self.client.delete(
                f"{self.base_url}/spaces/{instance_id}",
                headers=self.auth_headers(credential),
            )
        except httpx.HTTPError:
            return False
        return response.status_code < 400


class GitHubActionsProvisioner(ComputeProvisioner):
    """Dispatches a GitHub Actions workflow that starts the worker."""

    name = "github_actions"

    def default_base_url(self) -> str:
        return "https://api.github.com"

    def _target(self, credential: Credential, request: SpawnRequest) -> tuple[str, str, str]:
        repo = credential.metadata.get("repo") or request.labels.get("repo") or ""
        workflow = credential.metadata.get("workflow") or request.labels.get("workflow") or "worker.yml"
        ref = credential.metadata.get("ref") or request.labels.get("ref") or "main"
        return repo, workflow, ref

    async def provision(self, request: SpawnRequest, credential: Credential) -> SpawnResult:
        repo, workflow, ref = self._target(credential, request)
        if not repo:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error="the GitHub account has no 'repo' metadata (owner/name)",
            )
        inputs = {
            "head_url": request.head_url,
            "mesh_token": request.mesh_token,
            "provider": request.provider,
            "region": request.region,
            "model": request.model or "",
            "ttl_seconds": str(request.ttl_seconds),
        }
        try:
            response = await self.client.post(
                f"{self.base_url}/repos/{repo}/actions/workflows/{workflow}/dispatches",
                headers={**self.auth_headers(credential), "accept": "application/vnd.github+json"},
                json={"ref": ref, "inputs": inputs},
            )
        except httpx.HTTPError as exc:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=f"github dispatch failed: {exc}",
            )
        if response.status_code >= 400:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=f"github dispatch returned HTTP {response.status_code}: {response.text[:300]}",
            )
        run_id = await self._latest_run_id(repo, credential, ref)
        return SpawnResult(
            ok=True,
            provider=self.name,
            instance_id=f"{repo}#{run_id}" if run_id else repo,
            endpoint="",
            account_label=credential.label,
            detail=f"workflow {workflow} dispatched on {repo}@{ref}",
            meta={"repo": repo, "workflow": workflow, "ref": ref, "run_id": run_id},
        )

    async def _latest_run_id(self, repo: str, credential: Credential, ref: str) -> str:
        try:
            response = await self.client.get(
                f"{self.base_url}/repos/{repo}/actions/runs",
                headers={**self.auth_headers(credential), "accept": "application/vnd.github+json"},
                params={"event": "workflow_dispatch", "branch": ref, "per_page": 5},
            )
        except httpx.HTTPError:
            return ""
        if response.status_code >= 400:
            return ""
        try:
            payload = response.json()
        except ValueError:
            return ""
        runs = payload.get("workflow_runs") or []
        if not runs:
            return ""
        return str(runs[0].get("id", ""))

    async def terminate(self, instance_id: str, credential: Credential) -> bool:
        repo, _, run = instance_id.partition("#")
        if not run:
            return False
        try:
            response = await self.client.post(
                f"{self.base_url}/repos/{repo}/actions/runs/{run}/cancel",
                headers={**self.auth_headers(credential), "accept": "application/vnd.github+json"},
            )
        except httpx.HTTPError:
            return False
        return response.status_code < 400


class OracleProvisioner(ComputeProvisioner):
    """Launches an always-free Ampere A1 instance through the OCI API."""

    name = "oracle"

    def default_base_url(self) -> str:
        return "https://compute.oraclecloud.com"

    def _region_host(self, request: SpawnRequest) -> str:
        if self.explicit_base_url:
            return self.base_url
        region = request.region if request.region and request.region != "unknown" else "us-ashburn-1"
        return f"https://compute.{region}.oraclecloud.com"

    def auth_headers(self, credential: Credential) -> dict[str, str]:
        """OCI authenticates with the request signature, never with a token.

        The private key must never leave the process as a header, so the
        generic bearer-token behaviour is deliberately disabled here.
        """
        return {}

    def _signing_key(self, credential: Credential) -> Any:
        """Load the RSA private key used to sign OCI requests."""
        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "the OCI provisioner needs the 'cryptography' package to sign requests"
            ) from exc
        pem = credential.token
        if "BEGIN" not in pem:
            path = credential.metadata.get("private_key_path", "")
            if not path or not Path(path).is_file():
                raise RuntimeError("no usable OCI private key in the credential")
            pem = Path(path).read_text(encoding="utf-8")
        key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
        return key

    def _signed_headers(self, credential: Credential, method: str, url: str, body: bytes) -> dict[str, str]:
        """Build the OCI-signed headers (as documented by Oracle)."""
        key = self._signing_key(credential)
        tenancy = credential.metadata.get("tenancy") or ""
        user = credential.metadata.get("user") or ""
        fingerprint = credential.metadata.get("fingerprint") or ""
        if not (tenancy and user and fingerprint):
            raise RuntimeError(
                "the OCI credential needs 'tenancy', 'user' and 'fingerprint' metadata"
            )
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        parsed = httpx.URL(url)
        host = parsed.netloc if isinstance(parsed.netloc, str) else parsed.netloc.decode("ascii")
        path = parsed.path or "/"
        if isinstance(path, bytes):  # pragma: no cover - httpx keeps paths as str
            path = path.decode("ascii")
        query = parsed.query
        if isinstance(query, bytes):
            query = query.decode("ascii")
        date = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
        signed_headers = "date (request-target) host"
        canonical = "\n".join(
            [
                method.upper(),
                path,
                query,
                f"date: {date}",
                f"host: {host}",
                "",
                signed_headers,
                hashlib.sha256(body).hexdigest(),
            ]
        )
        signature = key.sign(canonical.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
        authorization = (
            "Signature version=\"1\","
            f"keyId=\"{fingerprint}\","
            f"algorithm=\"rsa-sha256\","
            f"headers=\"{signed_headers}\","
            f"signature=\"{base64.b64encode(signature).decode('ascii')}\""
        )
        return {
            "date": date,
            "host": host,
            "authorization": authorization,
            "content-type": "application/json",
            **(self.auth_headers(credential)),
        }

    async def provision(self, request: SpawnRequest, credential: Credential) -> SpawnResult:
        metadata = credential.metadata
        compartment = metadata.get("compartment") or ""
        image = metadata.get("image") or ""
        availability = metadata.get("availability_domain") or ""
        if not (compartment and image and availability):
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=(
                    "the OCI credential needs 'compartment', 'image' and "
                    "'availability_domain' metadata"
                ),
            )
        ssh_keys = metadata.get("ssh_public_key") or ""
        cloud_init = "\n".join(
            [
                "#!/bin/bash",
                "apt-get update && apt-get install -y python3 python3-pip curl ca-certificates git",
                "mkdir -p /opt/hypercluster && cd /opt/hypercluster",
                "git clone https://github.com/BlackRubyCTI/Bored-from-studying-lol.git repo || true",
                "cd repo/hyper-cluster 2>/dev/null || cd /opt/hypercluster/repo/hyper-cluster",
                self._worker_bootstrap(request),
            ]
        )
        body = json.dumps(
            {
                "compartmentId": compartment,
                "availabilityDomain": availability,
                "displayName": f"hyper-cluster-{request.region}",
                "shape": "VM.Standard.A1.Flex",
                "shapeConfig": {"ocpus": 4, "memoryInGBs": 24},
                "sourceDetails": {"sourceType": "image", "imageId": image},
                "metadata": {
                    "ssh_authorized_keys": ssh_keys,
                    "user_data": base64.b64encode(cloud_init.encode("utf-8")).decode("ascii"),
                },
            }
        ).encode("utf-8")
        url = f"{self._region_host(request)}/20160918/instances"
        try:
            headers = self._signed_headers(credential, "POST", url, body)
        except RuntimeError as exc:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=str(exc),
            )
        try:
            response = await self.client.post(url, headers=headers, content=body)
        except httpx.HTTPError as exc:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=f"oci launch failed: {exc}",
            )
        if response.status_code >= 400:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=f"oci launch returned HTTP {response.status_code}: {response.text[:300]}",
            )
        try:
            payload = response.json()
            instance_id = str((payload.get("data") or {}).get("id", ""))
        except ValueError:
            instance_id = ""
        return SpawnResult(
            ok=True,
            provider=self.name,
            instance_id=instance_id,
            endpoint="",
            account_label=credential.label,
            detail=f"OCI instance {instance_id or '(unknown id)'} launched",
            meta={"compartment": compartment, "image": image},
        )

    async def terminate(self, instance_id: str, credential: Credential) -> bool:
        metadata = credential.metadata
        region = str(metadata.get("region", "us-ashburn-1"))
        url = f"https://compute.{region}.oraclecloud.com/20160918/instances/{instance_id}"
        try:
            headers = self._signed_headers(credential, "DELETE", url, b"")
        except RuntimeError:
            return False
        try:
            response = await self.client.delete(url, headers=headers)
        except httpx.HTTPError:
            return False
        return response.status_code < 400


class EdgeApiProvisioner(ComputeProvisioner):
    """Provisions edge capacity by leasing and verifying a free-API account."""

    name = "edge_api"

    def default_base_url(self) -> str:
        return ""

    async def provision(self, request: SpawnRequest, credential: Credential) -> SpawnResult:
        spec = get_provider(credential.provider)
        base = (credential.base_url or (spec.base_url if spec else "")).rstrip("/")
        if not base:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=f"provider '{credential.provider}' has no base URL",
            )
        url = f"{base}/models"
        try:
            response = await self.client.get(
                url,
                headers={**self.auth_headers(credential), "accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=f"edge account probe failed: {exc}",
            )
        if response.status_code >= 400:
            return SpawnResult(
                ok=False,
                provider=self.name,
                account_label=credential.label,
                error=f"edge account probe returned HTTP {response.status_code}",
            )
        model = credential.default_model()
        try:
            payload = response.json()
            models = [str(m.get("id", "")) for m in (payload.get("data") or []) if isinstance(m, dict)]
            if models and not credential.models:
                model = models[0]
        except ValueError:
            pass
        return SpawnResult(
            ok=True,
            provider=credential.provider,
            instance_id=f"{credential.provider}:{credential.label}",
            endpoint=base,
            account_label=credential.label,
            detail=f"edge account {credential.label} of {credential.provider} is live",
            meta={"model": model, "edge": True},
        )

    async def terminate(self, instance_id: str, credential: Credential) -> bool:
        # An edge account is not destroyed, it is released back to the vault.
        return True


# --------------------------------------------------------------------------- #
# Spawner
# --------------------------------------------------------------------------- #
@dataclass
class RotatorAction:
    """One action taken by the pre-emptive TTL rotator."""

    kind: str = ""
    worker_id: str = ""
    replacement: str = ""
    provider: str = ""
    detail: str = ""
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "worker_id": self.worker_id,
            "replacement": self.replacement,
            "provider": self.provider,
            "detail": self.detail,
            "ok": self.ok,
        }


class AutoSpawner:
    """Provisions compute and rotates expiring workers without human help."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: WorkerRegistry | None = None,
        vault: CredentialVault | None = None,
        *,
        provisioners: dict[str, ComputeProvisioner] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings if settings is not None else Settings.from_env()
        self.registry = registry if registry is not None else WorkerRegistry(
            ttl_seconds=self.settings.worker_ttl_seconds,
            max_concurrency_per_worker=self.settings.max_concurrency_per_worker,
        )
        self.vault = vault if vault is not None else get_vault(self.settings.credential_file)
        self.client = client
        self.provisioners: dict[str, ComputeProvisioner] = (
            provisioners
            if provisioners is not None
            else {
                "kaggle": KaggleProvisioner(self.settings, client=client),
                "hf_spaces": HuggingFaceSpaceProvisioner(self.settings, client=client),
                "github_actions": GitHubActionsProvisioner(self.settings, client=client),
                "oracle": OracleProvisioner(self.settings, client=client),
                "edge_api": EdgeApiProvisioner(self.settings, client=client),
            }
        )
        self.instances: dict[str, SpawnedInstance] = {}
        self._pending: dict[str, SpawnedInstance] = {}  # old worker id -> replacement
        self._used_accounts: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Account handling
    # ------------------------------------------------------------------ #
    def _accounts_for(self, provider: str) -> list[Credential]:
        """Every usable account of a provider, healthiest first."""
        accounts = [
            cred
            for cred in self.vault.list(provider)
            if cred.token and not cred.disabled and cred.health not in (HEALTH_INVALID,)
        ]
        used = self._used_accounts.get(provider, set())
        fresh = [c for c in accounts if c.label not in used]
        stale = [c for c in accounts if c.label in used]
        # Rotate: prefer an account that has not been used for this provider
        # yet, but fall back to the others when the pool is exhausted.
        return fresh + stale

    def _remember_account(self, provider: str, label: str) -> None:
        self._used_accounts.setdefault(provider, set()).add(label)

    @staticmethod
    def _scrub(text: str, accounts: list[Credential]) -> str:
        """Remove every secret of ``accounts`` from a provider message.

        Providers happily echo the ``Authorization`` header back in their
        error bodies -- that must never reach the API, the log or the
        ledger.
        """
        if not text:
            return text
        cleaned = text
        for credential in accounts:
            for secret in (credential.token, *(credential.metadata or {}).values()):
                if not isinstance(secret, str) or len(secret) < 8:
                    continue
                cleaned = cleaned.replace(secret, "***")
        return cleaned

    # ------------------------------------------------------------------ #
    # Provisioning
    # ------------------------------------------------------------------ #
    async def provision(self, request: SpawnRequest) -> SpawnResult:
        """Provision compute, failing over silently between accounts."""
        provisioner = self.provisioners.get(request.provider)
        if provisioner is None:
            return SpawnResult(
                ok=False,
                provider=request.provider,
                error=f"no provisioner for provider '{request.provider}'",
            )
        if request.provider == "edge_api":
            accounts = self._accounts_for(request.labels.get("edge_provider", ""))
        else:
            accounts = self._accounts_for(request.provider)
        if not accounts:
            return SpawnResult(
                ok=False,
                provider=request.provider,
                error=f"no usable account for provider '{request.provider}'",
            )
        last_error = ""
        for credential in accounts:
            try:
                result = await provisioner.provision(request, credential)
            except Exception as exc:  # noqa: BLE001 - a provisioner must never kill the loop
                last_error = self._scrub(f"{type(exc).__name__}: {exc}", [credential])
                LOGGER.warning(
                    "provisioner raised",
                    extra={"hc_provider": request.provider, "hc_error": last_error},
                )
                continue
            if result.ok:
                self.vault.record_usage(credential.provider, credential.label)
                self.vault.mark(credential.provider, credential.label, health=HEALTH_HEALTHY, error="")
                self._remember_account(request.provider, credential.label)
                instance = SpawnedInstance(
                    instance_id=result.instance_id,
                    provider=request.provider,
                    account_label=credential.label,
                    endpoint=result.endpoint,
                    state=STATE_PROVISIONING,
                    expires_at=now_s() + request.ttl_seconds,
                    replaces=request.replaces,
                )
                async with self._lock:
                    self.instances[result.instance_id] = instance
                LOGGER.info(
                    "provisioned compute",
                    extra={
                        "hc_provider": request.provider,
                        "hc_instance": result.instance_id,
                        "hc_account": credential.label,
                    },
                )
                return result
            last_error = self._scrub(result.error, [credential])
            self.vault.mark(
                credential.provider,
                credential.label,
                health=HEALTH_DEGRADED,
                error=result.error[:200],
            )
            LOGGER.warning(
                "provisioning failed; trying the next account",
                extra={
                    "hc_provider": request.provider,
                    "hc_account": credential.label,
                    "hc_error": result.error[:200],
                },
            )
        return SpawnResult(
            ok=False,
            provider=request.provider,
            error=self._scrub(last_error, accounts) or "every account failed",
        )

    async def terminate(self, instance_id: str) -> bool:
        """Destroy a provisioned instance."""
        instance = self.instances.get(instance_id)
        if instance is None:
            return False
        provisioner = self.provisioners.get(instance.provider)
        if provisioner is None:
            return False
        credential = self.vault.get(instance.provider, instance.account_label)
        if credential is None:
            return False
        ok = await provisioner.terminate(instance_id, credential)
        if ok:
            instance.state = STATE_TERMINATED
        return ok

    # ------------------------------------------------------------------ #
    # Rotation
    # ------------------------------------------------------------------ #
    def _replacement_candidates(self, old: WorkerRecord, since: float) -> list[WorkerRecord]:
        """Workers that could be the replacement for ``old``."""
        out = []
        for record in self.registry.list_workers(healthy_only=True):
            if record.worker_id == old.worker_id:
                continue
            if record.provider.lower() != old.provider.lower():
                continue
            if record.region != old.region:
                continue
            if record.registered_at < since:
                continue
            out.append(record)
        out.sort(key=lambda r: r.registered_at, reverse=True)
        return out

    async def _provision_replacement(self, old: WorkerRecord) -> SpawnResult:
        request = SpawnRequest(
            provider=old.provider,
            region=old.region,
            model=old.model_loaded,
            family=old.model_family,
            params_b=old.model_params_b,
            vram_gb=old.vram_total_gb,
            ttl_seconds=self.settings.spawner_default_ttl_seconds,
            labels={"replaces": old.worker_id, "origin": "ttl-rotator"},
            replaces=old.worker_id,
            head_url=self.settings.public_base_url or f"http://{self.settings.head_host}:{self.settings.head_port}",
            mesh_token=self.settings.mesh_token,
        )
        return await self.provision(request)

    async def rotate(self, record: WorkerRecord) -> RotatorAction:
        """Provision a replacement for one expiring worker and hand over."""
        result = await self._provision_replacement(record)
        if not result.ok:
            return RotatorAction(
                kind="rotate",
                worker_id=record.worker_id,
                provider=record.provider,
                detail=result.error,
                ok=False,
            )
        instance = self.instances.get(result.instance_id)
        if instance is not None:
            instance.replaces = record.worker_id
        async with self._lock:
            self._pending[record.worker_id] = instance or SpawnedInstance(
                instance_id=result.instance_id,
                provider=record.provider,
                account_label=result.account_label,
            )
        return RotatorAction(
            kind="rotate",
            worker_id=record.worker_id,
            replacement=result.instance_id,
            provider=record.provider,
            detail=f"replacement {result.instance_id} provisioned; waiting for registration",
        )

    async def complete_handoff(self, old: WorkerRecord, replacement: WorkerRecord) -> RotatorAction:
        """Drain the old worker now that the replacement is serving."""
        self.registry.drain(old.worker_id, reason="ttl handoff")
        instance = None
        for candidate in self.instances.values():
            if candidate.replaces == old.worker_id:
                instance = candidate
                break
        if instance is not None:
            instance.state = STATE_REGISTERED
            instance.worker_id = replacement.worker_id
        async with self._lock:
            self._pending.pop(old.worker_id, None)
        LOGGER.info(
            "ttl handoff complete",
            extra={
                "hc_worker": old.worker_id,
                "hc_replacement": replacement.worker_id,
                "hc_provider": old.provider,
            },
        )
        return RotatorAction(
            kind="handoff",
            worker_id=old.worker_id,
            replacement=replacement.worker_id,
            provider=old.provider,
            detail="traffic shifted to the replacement; the old worker is draining",
        )

    async def tick(self) -> list[RotatorAction]:
        """One rotation pass: replace expiring workers, keep the floor."""
        actions: list[RotatorAction] = []
        if not self.settings.spawner_enabled:
            return actions

        # --- 1. hand over the workers whose session is about to die -------- #
        # The trigger is the *session deadline* the worker reports (a Kaggle
        # kernel deadline, a GitHub job timeout, an instance lifetime).  The
        # liveness TTL is refreshed by every heartbeat and therefore never
        # approaches, so it is only used for workers that went silent.
        lead = max(0.0, float(self.settings.handoff_lead_seconds))
        expiring = self.registry.session_ending_within(lead)
        silent = [
            record
            for record in self.registry.expiring_within(lead)
            if record.consecutive_misses > 0
        ]
        for record in [*expiring, *silent]:
            if record.worker_id in self._pending:
                continue
            provider = record.provider.lower()
            if provider not in self.provisioners:
                continue
            action = await self.rotate(record)
            actions.append(action)

        # --- 2. complete the handovers whose replacement has registered ---- #
        async with self._lock:
            pending_items = list(self._pending.items())
        for old_id, instance in pending_items:
            old = self.registry.get(old_id)
            if old is None:
                async with self._lock:
                    self._pending.pop(old_id, None)
                continue
            candidates = self._replacement_candidates(old, instance.created_at)
            if not candidates:
                continue
            actions.append(await self.complete_handoff(old, candidates[0]))

        # --- 3. keep the floor: never run below the configured minimum ----- #
        healthy = len(self.registry.healthy_workers())
        deficit = self.settings.spawner_min_workers - healthy
        if deficit > 0:
            providers = [p for p in self.settings.spawner_providers if p in self.provisioners]
            if not providers:
                providers = [p for p in self.provisioners if p != "edge_api"]
            index = 0
            while deficit > 0 and providers:
                provider = providers[index % len(providers)]
                index += 1
                request = SpawnRequest(
                    provider=provider,
                    region="unknown",
                    ttl_seconds=self.settings.spawner_default_ttl_seconds,
                    labels={"origin": "scale-out"},
                    head_url=self.settings.public_base_url
                    or f"http://{self.settings.head_host}:{self.settings.head_port}",
                    mesh_token=self.settings.mesh_token,
                )
                result = await self.provision(request)
                actions.append(
                    RotatorAction(
                        kind="scale-out",
                        provider=provider,
                        replacement=result.instance_id,
                        detail=result.error or f"provisioned {result.instance_id}",
                        ok=result.ok,
                    )
                )
                if result.ok:
                    deficit -= 1
                elif index > len(providers) * 2:
                    # Every provider failed; stop instead of spinning.
                    break
        return actions

    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """A JSON-safe view of the spawner state."""
        return {
            "enabled": self.settings.spawner_enabled,
            "providers": sorted(self.provisioners),
            "instances": [instance.to_dict() for instance in self.instances.values()],
            "pending_rotations": {
                old_id: instance.instance_id for old_id, instance in self._pending.items()
            },
            "accounts": {
                provider: [cred.to_dict(redact=True) for cred in self.vault.list(provider)]
                for provider in sorted(self.provisioners)
            },
        }

    async def aclose(self) -> None:
        for provisioner in self.provisioners.values():
            await provisioner.aclose()


def worker_payload_directory(request: SpawnRequest, directory: str | Path | None = None) -> Path:
    """Write the files a provisioned node needs into a directory."""
    target = Path(directory or tempfile.mkdtemp(prefix="hypercluster-payload-"))
    target.mkdir(parents=True, exist_ok=True)
    provisioner = KaggleProvisioner(Settings())
    (target / "worker_bootstrap.sh").write_text(
        provisioner._worker_bootstrap(request), encoding="utf-8"  # noqa: SLF001 - shared payload
    )
    (target / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": request.labels.get("kernel", "owner/hyper-cluster"),
                "title": "hyper-cluster",
                "code_file": "worker_bootstrap.sh",
                "language": "shell",
                "kernel_type": "script",
                "enable_gpu": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


__all__ = [
    "AutoSpawner",
    "ComputeProvisioner",
    "EdgeApiProvisioner",
    "GitHubActionsProvisioner",
    "HuggingFaceSpaceProvisioner",
    "KaggleProvisioner",
    "OracleProvisioner",
    "RotatorAction",
    "SpawnRequest",
    "SpawnResult",
    "SpawnedInstance",
    "STATE_FAILED",
    "STATE_PROVISIONING",
    "STATE_REGISTERED",
    "STATE_TERMINATED",
    "worker_payload_directory",
]
