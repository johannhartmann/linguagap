"""Dialogue generator using Gemini.

Generates realistic bilingual dialogue scenarios for E2E testing.
"""

import json
import os
from pathlib import Path

from tests.e2e.dialogues.templates import (
    SCENARIO_TYPES,
    TARGET_LANGUAGES,
    DialogueScenario,
    DialogueTurn,
)

# Gemini model for dialogue generation
GENERATOR_MODEL = "gemini-2.0-flash"


GENERATION_PROMPT = """Generate a realistic bilingual dialogue scenario for testing a speech-to-text translation system.

Requirements:
- Language pair: German (de) and {foreign_lang_name} ({foreign_lang_code})
- Scenario type: {scenario_type}
- Number of turns: 6-10 turns total
- Speaker 1 speaks German, Speaker 2 speaks {foreign_lang_name}
- Each speaker should have 3-5 turns
- The dialogue should be natural and realistic
- Include expected German translations for the {foreign_lang_name} text

Scenario type guidelines:
- customer_service: Simple Q&A about products/services, polite and formal register
- business_meeting: Technical discussion, formal language, professional terms
- casual_conversation: Informal chat, colloquial expressions, friendly tone
- code_switching: Include 1-2 instances where a speaker mixes both languages in one turn

Output format (JSON):
{{
    "name": "scenario_name",
    "description": "Brief description of the scenario",
    "speakers": {{
        "speaker_1": "German Speaker Name",
        "speaker_2": "{foreign_lang_name} Speaker Name"
    }},
    "turns": [
        {{
            "speaker_id": "speaker_1",
            "language": "de",
            "text": "German text here",
            "expected_translation": null
        }},
        {{
            "speaker_id": "speaker_2",
            "language": "{foreign_lang_code}",
            "text": "{foreign_lang_name} text here",
            "expected_translation": "German translation of the {foreign_lang_name} text"
        }}
    ],
    "expected_summary_topics": ["topic1", "topic2", "topic3"]
}}

Generate a complete, realistic dialogue now:"""


class DialogueGenerator:
    """Generates dialogue scenarios using Gemini."""

    def __init__(self, api_key: str | None = None):
        """Initialize the generator.

        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def generate_scenario(
        self,
        foreign_lang_code: str,
        scenario_type: str = "customer_service",
    ) -> DialogueScenario:
        """Generate a dialogue scenario.

        Args:
            foreign_lang_code: Language code (e.g., "uk", "ar", "tr")
            scenario_type: Type of scenario (see SCENARIO_TYPES)

        Returns:
            Generated DialogueScenario
        """
        foreign_lang_name = TARGET_LANGUAGES.get(foreign_lang_code, foreign_lang_code)

        prompt = GENERATION_PROMPT.format(
            foreign_lang_name=foreign_lang_name,
            foreign_lang_code=foreign_lang_code,
            scenario_type=scenario_type,
        )

        response = self.client.models.generate_content(
            model=GENERATOR_MODEL,
            contents=prompt,
        )

        # Parse JSON response
        if not response.text:
            raise ValueError("Empty response from Gemini")
        response_text = response.text.strip()
        # Handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (```json and ```)
            response_text = "\n".join(lines[1:-1])

        data = json.loads(response_text)

        # Convert to DialogueScenario
        turns = [
            DialogueTurn(
                speaker_id=t["speaker_id"],
                language=t["language"],
                text=t["text"],
                expected_translation=t.get("expected_translation"),
            )
            for t in data["turns"]
        ]

        return DialogueScenario(
            name=data["name"],
            description=data["description"],
            german_lang="de",
            foreign_lang=foreign_lang_code,
            speakers=data["speakers"],
            turns=turns,
            expected_summary_topics=data.get("expected_summary_topics", []),
        )

    def generate_all_scenarios(
        self,
        output_dir: str | Path = "tests/e2e/dialogues/samples",
    ) -> list[Path]:
        """Generate scenarios for all language pairs and scenario types.

        Args:
            output_dir: Directory to save generated YAML files

        Returns:
            List of paths to generated files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated_files = []

        for lang_code in TARGET_LANGUAGES:
            for scenario_type in SCENARIO_TYPES:
                print(f"Generating {lang_code} / {scenario_type}...")
                try:
                    scenario = self.generate_scenario(lang_code, scenario_type)
                    filename = f"{lang_code}_{scenario_type}.yaml"
                    filepath = output_path / filename
                    scenario.save_yaml(str(filepath))
                    generated_files.append(filepath)
                    print(f"  Saved: {filepath}")
                except Exception as e:
                    print(f"  Error: {e}")

        return generated_files


def main():
    """CLI entry point for generating dialogue scenarios."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate dialogue scenarios")
    parser.add_argument(
        "--language",
        "-l",
        choices=list(TARGET_LANGUAGES.keys()) + ["all"],
        default="all",
        help="Language code to generate for",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=SCENARIO_TYPES + ["all"],
        default="all",
        help="Scenario type to generate",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="tests/e2e/dialogues/samples",
        help="Output directory",
    )
    args = parser.parse_args()

    generator = DialogueGenerator()

    if args.language == "all" and args.scenario == "all":
        generator.generate_all_scenarios(args.output)
    else:
        languages = list(TARGET_LANGUAGES.keys()) if args.language == "all" else [args.language]
        scenarios = SCENARIO_TYPES if args.scenario == "all" else [args.scenario]

        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        for lang in languages:
            for scenario_type in scenarios:
                print(f"Generating {lang} / {scenario_type}...")
                scenario = generator.generate_scenario(lang, scenario_type)
                filename = f"{lang}_{scenario_type}.yaml"
                filepath = output_path / filename
                scenario.save_yaml(str(filepath))
                print(f"  Saved: {filepath}")


if __name__ == "__main__":
    main()
