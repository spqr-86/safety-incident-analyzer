"""A/B comparison of two eval runs.

Usage:
    python eval/compare.py eval/baselines/v7_baseline.json eval/baselines/v8_run.json

Выводит:
  - сравнение aggregate метрик (дельты)
  - список вопросов где completeness вырос / упал
  - рекомендацию: принять / отклонить V8
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_run(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _delta_str(a: float, b: float) -> str:
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def compare(path_a: Path, path_b: Path) -> None:
    run_a = load_run(path_a)
    run_b = load_run(path_b)

    label_a = run_a.get("label", path_a.stem)
    label_b = run_b.get("label", path_b.stem)
    m_a = run_a["metrics"]
    m_b = run_b["metrics"]

    print(f"\n{'=' * 65}")
    print(f"A/B Comparison: {label_a}  →  {label_b}")
    print(f"{'─' * 65}")
    print(f"{'Metric':<35} {'A':>10} {'B':>10} {'Delta':>10}")
    print(f"{'─' * 65}")

    metrics = [
        ("completeness_mean", "Completeness (all)", True),
        ("completeness_enumeration", "Completeness (enumeration)", True),
        ("abstain_rate", "Abstain rate", None),
        ("false_abstain_rate", "False abstain rate", False),
        ("correct_abstain_rate", "Correct abstain rate", True),
        ("avg_top_score", "Avg top_score", True),
        ("avg_passage_count", "Avg passage_count", None),
    ]

    wins, losses = 0, 0
    for key, label, higher_is_better in metrics:
        va = m_a.get(key, 0.0)
        vb = m_b.get(key, 0.0)
        d = vb - va
        delta = _delta_str(va, vb)
        if higher_is_better is True and d > 0.01:
            wins += 1
        elif higher_is_better is False and d < -0.01:
            wins += 1
        elif higher_is_better is True and d < -0.01:
            losses += 1
        elif higher_is_better is False and d > 0.01:
            losses += 1
        print(f"  {label:<33} {va:>10.4f} {vb:>10.4f} {delta:>10}")

    print(f"\n{'─' * 65}")
    print(f"  Wins: {wins}  |  Losses: {losses}")

    # ── Критические проверки ─────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("Critical checks:")
    checks = [
        (m_b.get("false_abstain_rate", 1.0) == 0.0, "False abstain rate = 0"),
        (m_b.get("correct_abstain_rate", 0.0) >= 1.0, "Correct abstain (OOS) = 100%"),
        (
            m_b.get("completeness_enumeration", 0.0)
            > m_a.get("completeness_enumeration", 0.0),
            "Enumeration completeness improved",
        ),
    ]
    all_pass = True
    for ok, label in checks:
        status = _pass_fail(ok)
        if not ok:
            all_pass = False
        print(f"  [{status}] {label}")

    # ── Per-question diff (completeness) ─────────────────────────────────────
    results_a = {r["question"]: r for r in run_a.get("results", [])}
    results_b = {r["question"]: r for r in run_b.get("results", [])}
    common = set(results_a) & set(results_b)

    improved, regressed = [], []
    for q in common:
        ca = results_a[q].get("completeness", 0.0)
        cb = results_b[q].get("completeness", 0.0)
        if cb - ca > 0.05:
            improved.append((q, ca, cb))
        elif ca - cb > 0.05:
            regressed.append((q, ca, cb))

    improved.sort(key=lambda x: x[2] - x[1], reverse=True)
    regressed.sort(key=lambda x: x[1] - x[2], reverse=True)

    if improved:
        print(f"\nImproved ({len(improved)}):")
        for q, ca, cb in improved[:5]:
            print(f"  +{cb - ca:.2f}  [{ca:.2f}→{cb:.2f}]  {q[:60]}")
    if regressed:
        print(f"\nRegressed ({len(regressed)}):")
        for q, ca, cb in regressed[:5]:
            print(f"  -{ca - cb:.2f}  [{ca:.2f}→{cb:.2f}]  {q[:60]}")

    print(f"\n{'=' * 65}")
    verdict = (
        "RECOMMEND ACCEPT" if (all_pass and wins >= losses) else "RECOMMEND REJECT"
    )
    print(f"Verdict: {verdict} ({label_b})")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval/compare.py <run_a.json> <run_b.json>")
        sys.exit(1)
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
