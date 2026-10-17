"""Plot the six recorded official Public scores for the retrospective README."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "docs/evidence" / name).read_text(encoding="utf-8"))


def main():
    first = read("2026-10-07-first-submission.json")["official_submission"]
    second = read("2026-10-07-score-research.json")["official_v2"]
    sources = [
        (1, "DINOv3 baseline", first["public_score"], "2026-10-07-first-submission.json"),
        (2, "Gallery reranking", second["score"], "2026-10-07-score-research.json"),
    ]
    for version, name, title in (
        (3, "2026-10-07-third-submission.json", "TransOSS"),
        (4, "2026-10-08-fourth-submission.json", "Query-cohort matching"),
        (5, "2026-10-08-fifth-submission.json", "Learned local pairing"),
        (6, "2026-10-11-sixth-score.json", "Two-scale experts"),
    ):
        sources.append((version, title, read(name)["official_public_score"], name))
    versions = np.array([r[0] for r in sources])
    scores = np.array([r[2] for r in sources])
    best = int(scores.argmax())
    assert int(versions[best]) == 5
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    fig.suptitle("CUNSL 2026 / Track 6 — Six recorded Public submissions", fontsize=17, y=.96)
    ax.plot(versions, scores, color="#536d89", linewidth=2.5, marker="o", markersize=8, label="Submitted score")
    ax.plot(versions, np.maximum.accumulate(scores), color="#259584", linewidth=1.5, linestyle="--", label="Best recorded score")
    ax.scatter(versions[best], scores[best], color="#158272", s=125, zorder=5)
    for v, score in zip(versions, scores):
        offset = -22 if v in (2, 6) else 12
        ax.annotate(f"{score:.6f}", (v, score), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=11, color="#158272" if v == 5 else "#283548", fontweight="bold" if v == 5 else "normal")
    ax.set_xticks(versions, ["V1\nDINOv3", "V2\nGallery rerank", "V3\nTransOSS", "V4\nCohort matching", "V5\nLocal pairing", "V6\nScale experts"], fontsize=10)
    ax.set(ylabel="Official Public score", ylim=(.36, .79), xlim=(.65, 6.35))
    ax.yaxis.grid(alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.text(.025, .018, "Recorded receipts only; last submission: 2026-10-11. No final-rank or award claim. Retrospective prepared 2026-10-18.", fontsize=9, color="#586478")
    fig.tight_layout(rect=(0, .06, 1, .91))
    public = ROOT / "docs/figures/2026-10-18"
    private = ROOT / "local/readme-summary-20261018"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    fig.savefig(public / "submission-history.svg", facecolor="white")
    fig.savefig(private / "submission-history.png", dpi=150, facecolor="white")
    plt.close(fig)
    evidence = {
        "prepared_at": "2026-10-18", "scope": "Six recorded official Public scores, not final standings",
        "best_version": int(versions[best]), "best_public_score": float(scores[best]),
        "rows": [{"version": v, "method": method, "public_score": score, "receipt": path} for v, method, score, path in sources],
        "V3_to_V4_gain": float(scores[3] - scores[2]), "V4_to_V5_gain": float(scores[4] - scores[3]),
        "V5_to_V6_gain": float(scores[5] - scores[4]),
        "final_rank_verified": False,
        "readme_reference": "https://github.com/Puiching-Memory/TAAC_2026",
        "reference_scope": "Centered title, status callout, navigation, feature summary, experiment entry points; no copied community or rank claims",
    }
    (ROOT / "docs/evidence/2026-10-18-submission-summary.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_version": evidence["best_version"], "best_public_score": evidence["best_public_score"], "figure": str(public / "submission-history.svg")}))


if __name__ == "__main__":
    main()
