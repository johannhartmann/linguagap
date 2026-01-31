"""Dialogue templates and dataclasses for E2E tests.

Defines the structure for bilingual dialogue scenarios.
"""

from dataclasses import dataclass, field

import yaml


@dataclass
class DialogueTurn:
    """A single turn in a dialogue.

    Attributes:
        speaker_id: Identifier for the speaker (e.g., "speaker_1", "speaker_2")
        language: Language code (e.g., "de", "uk", "ar")
        text: The spoken text in the original language
        expected_translation: Optional expected translation to German (for non-German)
    """

    speaker_id: str
    language: str
    text: str
    expected_translation: str | None = None


@dataclass
class DialogueScenario:
    """A complete dialogue scenario for testing.

    Attributes:
        name: Scenario name/identifier
        description: Brief description of the scenario
        german_lang: Always "de" for German
        foreign_lang: The non-German language code
        speakers: Dict mapping speaker_id to display name
        turns: List of dialogue turns
        expected_summary_topics: Topics that should appear in the summary
    """

    name: str
    description: str
    german_lang: str
    foreign_lang: str
    speakers: dict[str, str]
    turns: list[DialogueTurn]
    expected_summary_topics: list[str] = field(default_factory=list)

    def to_yaml(self) -> str:
        """Serialize scenario to YAML string."""
        data = {
            "name": self.name,
            "description": self.description,
            "german_lang": self.german_lang,
            "foreign_lang": self.foreign_lang,
            "speakers": self.speakers,
            "turns": [
                {
                    "speaker_id": t.speaker_id,
                    "language": t.language,
                    "text": t.text,
                    "expected_translation": t.expected_translation,
                }
                for t in self.turns
            ],
            "expected_summary_topics": self.expected_summary_topics,
        }
        return yaml.dump(data, allow_unicode=True, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "DialogueScenario":
        """Load scenario from YAML string."""
        data = yaml.safe_load(yaml_str)
        turns = [
            DialogueTurn(
                speaker_id=t["speaker_id"],
                language=t["language"],
                text=t["text"],
                expected_translation=t.get("expected_translation"),
            )
            for t in data["turns"]
        ]
        return cls(
            name=data["name"],
            description=data["description"],
            german_lang=data["german_lang"],
            foreign_lang=data["foreign_lang"],
            speakers=data["speakers"],
            turns=turns,
            expected_summary_topics=data.get("expected_summary_topics", []),
        )

    @classmethod
    def from_yaml_file(cls, path: str) -> "DialogueScenario":
        """Load scenario from YAML file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    def save_yaml(self, path: str) -> None:
        """Save scenario to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())


# Scenario types for generation
SCENARIO_TYPES = [
    "customer_service",  # Simple Q&A, polite register
    "business_meeting",  # Technical terms, formal
    "casual_conversation",  # Colloquial, informal
    "code_switching",  # Mixed language mid-sentence (stress test)
]

# Target language pairs (German + foreign)
TARGET_LANGUAGES = {
    "uk": "Ukrainian",
    "sq": "Albanian",
    "fa": "Farsi/Persian",
    "ar": "Arabic",
    "tr": "Turkish",
}
