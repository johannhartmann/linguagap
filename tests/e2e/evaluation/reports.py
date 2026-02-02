"""Report generation for E2E test results.

Generates JSON and HTML reports from evaluation results.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from tests.e2e.evaluation.judge import EvaluationResult


@dataclass
class ScenarioReport:
    """Report for a single test scenario.

    Attributes:
        scenario_name: Name of the dialogue scenario
        foreign_lang: Non-German language tested
        scenario_type: Type of scenario (customer_service, etc.)
        transcription_score: Transcription evaluation result
        translation_score: Translation evaluation result
        summary_score: Summary evaluation result
        language_detection_score: Language detection evaluation result
        speaker_diarization_score: Speaker diarization evaluation result
        errors: List of errors encountered
        duration_sec: Test duration in seconds
    """

    scenario_name: str
    foreign_lang: str
    scenario_type: str
    transcription_score: EvaluationResult | None = None
    translation_score: EvaluationResult | None = None
    summary_score: EvaluationResult | None = None
    language_detection_score: EvaluationResult | None = None
    speaker_diarization_score: EvaluationResult | None = None
    errors: list[str] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def overall_score(self) -> float:
        """Calculate average score across all evaluations."""
        scores = []
        for result in [
            self.transcription_score,
            self.translation_score,
            self.summary_score,
            self.language_detection_score,
            self.speaker_diarization_score,
        ]:
            if result is not None:
                scores.append(result.score)
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def passed(self) -> bool:
        """Check if all evaluations passed (score >= 3)."""
        for result in [
            self.transcription_score,
            self.translation_score,
            self.summary_score,
            self.language_detection_score,
            self.speaker_diarization_score,
        ]:
            if result is not None and not result.passed():
                return False
        return True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_name": self.scenario_name,
            "foreign_lang": self.foreign_lang,
            "scenario_type": self.scenario_type,
            "transcription": asdict(self.transcription_score) if self.transcription_score else None,
            "translation": asdict(self.translation_score) if self.translation_score else None,
            "summary": asdict(self.summary_score) if self.summary_score else None,
            "language_detection": (
                asdict(self.language_detection_score) if self.language_detection_score else None
            ),
            "speaker_diarization": (
                asdict(self.speaker_diarization_score) if self.speaker_diarization_score else None
            ),
            "overall_score": self.overall_score,
            "passed": self.passed,
            "errors": self.errors,
            "duration_sec": self.duration_sec,
        }


@dataclass
class TestReport:
    """Complete E2E test report.

    Attributes:
        scenarios: List of scenario reports
        timestamp: When the test was run
        environment: Environment information
    """

    scenarios: list[ScenarioReport] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: dict = field(default_factory=dict)

    @property
    def total_scenarios(self) -> int:
        """Total number of scenarios tested."""
        return len(self.scenarios)

    @property
    def passed_scenarios(self) -> int:
        """Number of scenarios that passed."""
        return sum(1 for s in self.scenarios if s.passed)

    @property
    def failed_scenarios(self) -> int:
        """Number of scenarios that failed."""
        return self.total_scenarios - self.passed_scenarios

    @property
    def overall_pass_rate(self) -> float:
        """Overall pass rate as a percentage."""
        if not self.scenarios:
            return 0.0
        return (self.passed_scenarios / self.total_scenarios) * 100

    @property
    def average_score(self) -> float:
        """Average score across all scenarios."""
        scores = [s.overall_score for s in self.scenarios if s.overall_score > 0]
        return sum(scores) / len(scores) if scores else 0.0

    def add_scenario(self, scenario: ScenarioReport) -> None:
        """Add a scenario report."""
        self.scenarios.append(scenario)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "environment": self.environment,
            "summary": {
                "total_scenarios": self.total_scenarios,
                "passed": self.passed_scenarios,
                "failed": self.failed_scenarios,
                "pass_rate": self.overall_pass_rate,
                "average_score": self.average_score,
            },
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    def save_json(self, path: str | Path) -> Path:
        """Save report as JSON.

        Args:
            path: Output file path

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def save_html(self, path: str | Path) -> Path:
        """Save report as HTML.

        Args:
            path: Output file path

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        html = self._generate_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def _generate_html(self) -> str:
        """Generate HTML report."""

        # Score color helper
        def score_color(score: float) -> str:
            if score >= 4:
                return "#22c55e"  # green
            elif score >= 3:
                return "#eab308"  # yellow
            else:
                return "#ef4444"  # red

        # Build scenario rows
        scenario_rows = []
        for s in self.scenarios:
            status = "PASS" if s.passed else "FAIL"
            status_color = "#22c55e" if s.passed else "#ef4444"

            scores_html = []
            for name, result in [
                ("Transcription", s.transcription_score),
                ("Translation", s.translation_score),
                ("Summary", s.summary_score),
                ("Language", s.language_detection_score),
                ("Speakers", s.speaker_diarization_score),
            ]:
                if result:
                    color = score_color(result.score)
                    scores_html.append(f'<span style="color:{color}">{name}: {result.score}</span>')

            scenario_rows.append(f"""
            <tr>
                <td>{s.scenario_name}</td>
                <td>{s.foreign_lang}</td>
                <td>{s.scenario_type}</td>
                <td style="color:{status_color};font-weight:bold">{status}</td>
                <td style="color:{score_color(s.overall_score)}">{s.overall_score:.1f}</td>
                <td>{" | ".join(scores_html)}</td>
            </tr>
            """)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Linguagap E2E Test Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1f2937; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ background: #f9fafb; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ color: #6b7280; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f9fafb; font-weight: 600; }}
        tr:hover {{ background: #f9fafb; }}
        .timestamp {{ color: #6b7280; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Linguagap E2E Test Report</h1>
        <p class="timestamp">Generated: {self.timestamp}</p>

        <div class="summary">
            <div class="stat">
                <div class="stat-value">{self.total_scenarios}</div>
                <div class="stat-label">Total Scenarios</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color:#22c55e">{self.passed_scenarios}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color:#ef4444">{self.failed_scenarios}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.overall_pass_rate:.0f}%</div>
                <div class="stat-label">Pass Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color:{score_color(self.average_score)}">{self.average_score:.1f}</div>
                <div class="stat-label">Avg Score</div>
            </div>
        </div>

        <h2>Scenario Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Language</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
                {"".join(scenario_rows)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
