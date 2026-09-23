"""Small, dependency-free checks for the public research repository."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        "README.md",
        "MODEL_CARD.md",
        "LICENSE",
        "configs/taj_recipe_v1.json",
        "docs/DATASET_METHOD.md",
        "docs/EXPERIMENT_JOURNEY.md",
        "docs/TIMELINE.md",
        "docs/RESULTS.md",
        "docs/REPRODUCIBILITY.md",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    assert not missing, f"Missing required files: {missing}"

    config = json.loads((ROOT / "configs/taj_recipe_v1.json").read_text())
    assert config["model"] == "models/qwen3-4b-base"
    assert config["iters"] == 375
    assert config["lora_parameters"] == {
        "rank": 16,
        "scale": 32.0,
        "dropout": 0.0,
    }

    markdown_files = list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md"))
    link_re = re.compile(r"\[[^]]+\]\((?!https?://|#)([^)]+)\)")
    broken: list[str] = []
    for document in markdown_files:
        for target in link_re.findall(document.read_text()):
            target = target.split("#", 1)[0]
            if target and not (document.parent / target).resolve().exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not broken, "Broken local Markdown links:\n" + "\n".join(broken)

    forbidden_suffixes = {".safetensors", ".gguf", ".onnx"}
    forbidden = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix in forbidden_suffixes
    ]
    assert not forbidden, f"Large model artifacts must stay on Hugging Face: {forbidden}"
    print(f"Repository checks passed: {len(markdown_files)} Markdown files checked.")


if __name__ == "__main__":
    main()
