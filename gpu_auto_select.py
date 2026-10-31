#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-select and reserve an idle GPU for one process.

The selection is coordination-safe across local processes via file locks, so
multiple concurrent runs do not all grab the same apparently-idle card.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GpuLease:
    requested: str
    local_device: str
    physical_gpu: int | None
    visible_devices: str | None
    source: str
    min_free_gb: float
    lock_path: str | None = None
    _lock_handle: object | None = None

    def activate_env(self) -> None:
        if self.requested in {"cpu", "none"}:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return
        if self.visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.visible_devices

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "local_device": self.local_device,
            "physical_gpu": self.physical_gpu,
            "visible_devices": self.visible_devices,
            "source": self.source,
            "min_free_gb": self.min_free_gb,
            "lock_path": self.lock_path,
        }

    def close(self) -> None:
        handle = self._lock_handle
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass
        self._lock_handle = None

    def __enter__(self) -> "GpuLease":
        self.activate_env()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _parse_visible_candidates() -> set[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if token.isdigit():
            out.add(int(token))
    return out or None


def _query_gpu_free_mb() -> list[tuple[int, int]]:
    last_err: Exception | None = None
    for _ in range(6):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            break
        except subprocess.CalledProcessError as exc:
            last_err = exc
            time.sleep(2.0)
    else:
        if last_err is not None:
            raise last_err
        raise RuntimeError("nvidia-smi query failed without exception detail")

    rows: list[tuple[int, int]] = []
    for line in result.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        if not parts[0].isdigit():
            continue
        try:
            rows.append((int(parts[0]), int(parts[1])))
        except Exception:
            continue
    return rows


def _open_lock(lock_dir: Path, gpu_id: int):
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"gpu{gpu_id}.lock"
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return str(path), handle
    except BlockingIOError:
        handle.close()
        return None, None


def acquire_gpu_lease(
    requested: str = "auto",
    min_free_gb: float = 8.0,
    wait_seconds: float = 600.0,
    poll_seconds: float = 10.0,
    lock_dir: str = "/tmp/llms_aes_gpu_locks",
) -> GpuLease:
    requested = str(requested).strip().lower()
    if requested in {"cpu", "none"}:
        return GpuLease(
            requested=requested,
            local_device="cpu",
            physical_gpu=None,
            visible_devices=None,
            source="explicit_cpu",
            min_free_gb=float(min_free_gb),
        )

    min_free_mb = int(round(float(min_free_gb) * 1024))
    allowed = _parse_visible_candidates()
    deadline = time.time() + max(0.0, float(wait_seconds))
    lock_root = Path(lock_dir)

    if requested.isdigit():
        gpu_id = int(requested)
        if allowed is not None and gpu_id not in allowed:
            visible = ",".join(str(x) for x in sorted(allowed))
            raise RuntimeError(f"Requested gpu={gpu_id} is not in CUDA_VISIBLE_DEVICES={visible}")
        while True:
            lock_path, handle = _open_lock(lock_root, gpu_id)
            if handle is not None:
                return GpuLease(
                    requested=requested,
                    local_device="cuda:0",
                    physical_gpu=gpu_id,
                    visible_devices=str(gpu_id),
                    source="explicit_gpu",
                    min_free_gb=float(min_free_gb),
                    lock_path=lock_path,
                    _lock_handle=handle,
                )
            if time.time() >= deadline:
                raise RuntimeError(f"Requested gpu={gpu_id} is locked after waiting {wait_seconds}s.")
            time.sleep(max(0.5, float(poll_seconds)))

    while True:
        rows = _query_gpu_free_mb()
        candidates: list[tuple[int, int]] = []
        for gpu_id, free_mb in rows:
            if allowed is not None and gpu_id not in allowed:
                continue
            if requested != "auto" and requested.isdigit() and gpu_id != int(requested):
                continue
            if requested == "auto" and free_mb < min_free_mb:
                continue
            candidates.append((gpu_id, free_mb))

        if requested == "auto":
            candidates.sort(key=lambda item: (-item[1], item[0]))
        else:
            candidates.sort(key=lambda item: item[0])

        for gpu_id, free_mb in candidates:
            lock_path, handle = _open_lock(lock_root, gpu_id)
            if handle is None:
                continue
            return GpuLease(
                requested=requested,
                local_device="cuda:0",
                physical_gpu=int(gpu_id),
                visible_devices=str(gpu_id),
                source=f"auto_free_mb={free_mb}" if requested == "auto" else "explicit_gpu",
                min_free_gb=float(min_free_gb),
                lock_path=lock_path,
                _lock_handle=handle,
            )

        if time.time() >= deadline:
            visible = "all" if allowed is None else ",".join(str(x) for x in sorted(allowed))
            raise RuntimeError(
                f"No GPU available for requested={requested} with min_free_gb={min_free_gb} "
                f"among visible={visible} after waiting {wait_seconds}s."
            )
        time.sleep(max(0.5, float(poll_seconds)))
