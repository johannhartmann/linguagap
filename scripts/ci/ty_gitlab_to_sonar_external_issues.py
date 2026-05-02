#!/usr/bin/env python3
"""Convert ty GitLab JSON output to SonarQube generic external issues format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TY_TO_SONAR_SEVERITY = {
    "info": "INFO",
    "minor": "MINOR",
    "major": "MAJOR",
    "critical": "CRITICAL",
    "blocker": "BLOCKER",
}

SEVERITY_RANK = {
    "INFO": 1,
    "MINOR": 2,
    "MAJOR": 3,
    "CRITICAL": 4,
    "BLOCKER": 5,
}

SEVERITY_TO_IMPACT = {
    "INFO": "INFO",
    "MINOR": "LOW",
    "MAJOR": "MEDIUM",
    "CRITICAL": "HIGH",
    "BLOCKER": "BLOCKER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ty GitLab JSON output to Sonar external issues JSON."
    )
    parser.add_argument("--input", required=True, help="Path to ty GitLab JSON report.")
    parser.add_argument(
        "--output", required=True, help="Path to Sonar external issues JSON report."
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Repository root directory used for file path normalization.",
    )
    parser.add_argument(
        "--service-root",
        default=None,
        help="Optional service root used when ty emits relative paths. "
        "Defaults to --repo-root for single-project layouts.",
    )
    return parser.parse_args()


def normalize_path(raw_path: str, repo_root: Path, service_root: Path) -> str | None:
    path = Path(raw_path)
    if path.is_absolute():
        candidate = path.resolve()
    else:
        candidate = (service_root / path).resolve()

    try:
        rel = candidate.relative_to(repo_root)
    except ValueError:
        return None
    return rel.as_posix()


def sonar_severity(ty_severity: str | None) -> str:
    if ty_severity is None:
        return "MAJOR"
    return TY_TO_SONAR_SEVERITY.get(ty_severity.lower(), "MAJOR")


def highest_severity(left: str, right: str) -> str:
    return left if SEVERITY_RANK[left] >= SEVERITY_RANK[right] else right


def to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def build_text_range(positions: dict[str, Any]) -> dict[str, int]:
    begin = positions.get("begin", {})
    end = positions.get("end", {})

    start_line = to_int(begin.get("line")) or 1
    start_column_raw = to_int(begin.get("column"))
    end_line = to_int(end.get("line"))
    end_column_raw = to_int(end.get("column"))

    text_range: dict[str, int] = {"startLine": start_line}

    if end_line is not None:
        text_range["endLine"] = end_line
    if start_column_raw is not None and start_column_raw > 0:
        text_range["startColumn"] = max(start_column_raw - 1, 0)
    if end_column_raw is not None and end_column_raw > 0:
        text_range["endColumn"] = max(end_column_raw - 1, 0)

    return text_range


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    repo_root = Path(args.repo_root).resolve()
    service_root = (
        Path(args.service_root).resolve() if args.service_root else repo_root
    )

    ty_issues = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(ty_issues, list):
        raise ValueError("Expected ty GitLab report to be a JSON array.")

    rules_by_id: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []

    for entry in ty_issues:
        if not isinstance(entry, dict):
            continue

        rule_id = str(entry.get("check_name") or "unknown-ty-rule")
        message = str(entry.get("description") or rule_id)
        severity = sonar_severity(entry.get("severity"))
        impact_severity = SEVERITY_TO_IMPACT[severity]

        location = entry.get("location", {})
        raw_path = location.get("path")
        if not isinstance(raw_path, str):
            continue

        file_path = normalize_path(
            raw_path, repo_root=repo_root, service_root=service_root
        )
        if file_path is None:
            continue

        positions = location.get("positions", {})
        text_range = build_text_range(positions if isinstance(positions, dict) else {})

        existing_rule = rules_by_id.get(rule_id)
        if existing_rule is None:
            rules_by_id[rule_id] = {
                "id": rule_id,
                "name": rule_id,
                "description": message,
                "engineId": "ty",
                "cleanCodeAttribute": "CONVENTIONAL",
                "type": "BUG",
                "severity": severity,
                "impacts": [
                    {
                        "softwareQuality": "RELIABILITY",
                        "severity": impact_severity,
                    }
                ],
            }
        else:
            merged = highest_severity(existing_rule["severity"], severity)
            existing_rule["severity"] = merged
            existing_rule["impacts"] = [
                {
                    "softwareQuality": "RELIABILITY",
                    "severity": SEVERITY_TO_IMPACT[merged],
                }
            ]

        issues.append(
            {
                "ruleId": rule_id,
                "primaryLocation": {
                    "message": message,
                    "filePath": file_path,
                    "textRange": text_range,
                },
            }
        )

    payload = {"rules": list(rules_by_id.values()), "issues": issues}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(issues)} issues and {len(rules_by_id)} rules to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
