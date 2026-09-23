"""Atomic MLX LoRA, optimizer, and random-state checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path: Path | str, value: Any) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def _encode(value: Any, arrays: dict[str, mx.array]) -> dict[str, Any]:
    if isinstance(value, mx.array):
        key = f"a{len(arrays):06d}"
        arrays[key] = value
        return {"array": key}
    if isinstance(value, dict):
        return {"dict": [[key, _encode(item, arrays)] for key, item in value.items()]}
    if isinstance(value, list):
        return {"list": [_encode(item, arrays) for item in value]}
    if isinstance(value, tuple):
        return {"tuple": [_encode(item, arrays) for item in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"value": value}
    raise TypeError(type(value))


def _decode(value: dict[str, Any], arrays: dict[str, mx.array]) -> Any:
    if "array" in value:
        return arrays[value["array"]]
    if "dict" in value:
        return {key: _decode(item, arrays) for key, item in value["dict"]}
    if "list" in value:
        return [_decode(item, arrays) for item in value["list"]]
    if "tuple" in value:
        return tuple(_decode(item, arrays) for item in value["tuple"])
    return value["value"]


def save_state(folder: Path, model: Any, optimizer: Any, metadata: dict[str, Any]) -> None:
    """Write a complete checkpoint and atomically make it visible."""
    temporary = folder.with_name(folder.name + ".incomplete")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    arrays: dict[str, mx.array] = {}
    numpy_state = np.random.get_state()
    tree = _encode(
        {"optimizer": optimizer.state, "rng": list(mx.random.state)}, arrays
    )
    mx.eval(model.trainable_parameters(), optimizer.state, mx.random.state)
    mx.save_safetensors(
        str(temporary / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    mx.save_safetensors(str(temporary / "state.safetensors"), arrays)

    document = {
        **metadata,
        "tree": tree,
        "numpy_rng": [
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "adapter_sha256": sha(temporary / "adapters.safetensors"),
        "state_sha256": sha(temporary / "state.safetensors"),
    }
    (temporary / "state.json").write_text(json.dumps(document))
    for path in temporary.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    assert not folder.exists(), folder
    temporary.replace(folder)
    atomic(folder.parent / "LATEST.json", {"path": folder.name, "step": metadata["step"]})


def restore_state(folder: Path, model: Any, optimizer: Any) -> dict[str, Any]:
    """Verify and restore LoRA, optimizer, MLX RNG, and NumPy RNG state."""
    metadata = json.loads((folder / "state.json").read_text())
    assert sha(folder / "adapters.safetensors") == metadata["adapter_sha256"]
    assert sha(folder / "state.safetensors") == metadata["state_sha256"]
    model.load_weights(str(folder / "adapters.safetensors"), strict=False)
    state = _decode(metadata["tree"], mx.load(str(folder / "state.safetensors")))
    optimizer.state = state["optimizer"]
    mx.random.state[0][...] = state["rng"][0]
    numpy_state = metadata["numpy_rng"]
    np.random.set_state(
        (
            numpy_state[0],
            np.array(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    mx.eval(model.trainable_parameters(), optimizer.state, mx.random.state)
    return metadata

