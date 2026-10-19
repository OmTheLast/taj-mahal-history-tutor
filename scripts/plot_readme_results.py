"""Plot published aggregate scores only; no inference or private evaluation data."""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "docs/figures/readme-results.json").read_text())
    rows = data["sections"]
    assert sum(r["base"] for r in rows) == 50
    assert sum(r["released"] for r in rows) == 60
    assert sum(r["total"] for r in rows) == 96
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="460" viewBox="0 0 960 460" role="img" aria-labelledby="title desc">',
           '<title id="title">Taj pilot: base versus released step100</title>',
           '<desc id="desc">Section scores on the frozen 96-question pilot. Supplied-note performance regresses.</desc>',
           '<rect width="960" height="460" fill="#fafaf8"/>',
           '<g font-family="Arial, sans-serif" fill="#20252b">',
           '<text x="28" y="36" font-size="23">Taj pilot: gains and regressions</text>',
           '<text x="28" y="65" font-size="15">Published aggregate scores • common 0–32 scale • base 50/96, released 60/96</text>']
    for tick in range(0, 33, 8):
        x = 230 + tick * 18
        svg += [f'<line x1="{x}" y1="95" x2="{x}" y2="385" stroke="#d9dde0"/>',
                f'<text x="{x}" y="410" text-anchor="middle" font-size="13">{tick}</text>']
    for i, row in enumerate(rows):
        y = 112 + i * 70
        svg.append(f'<text x="28" y="{y+18}" font-size="16">{escape(row["name"])}</text>')
        for j, (key, color) in enumerate([("base", "#8796a5"), ("released", "#176b75")]):
            yy = y + j * 25
            score = row[key]
            assert 0 <= score <= row["total"]
            svg += [f'<rect x="230" y="{yy}" width="{score*18}" height="19" fill="{color}"/>',
                    f'<text x="{238+score*18}" y="{yy+15}" font-size="14">{score}/{row["total"]}</text>']
    svg += ['<text x="28" y="445" font-size="14">Gray: unchanged base   Teal: released step100   |   Source: docs/RESULTS.md</text>', '</g></svg>']
    output = ROOT / "docs/figures/readme-results.svg"
    output.write_text("\n".join(svg) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
