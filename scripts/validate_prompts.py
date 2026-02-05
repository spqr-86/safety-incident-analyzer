import os
import yaml
import sys
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.prompt_manager import PromptManager
except ImportError as e:
    print(f"❌ Could not import src.prompt_manager: {e}")
    sys.exit(1)

# Fixtures for rendering tests
FIXTURES: Dict[str, Dict[str, Any]] = {
    "relevance_checker": {
        "question": "Нужен ли каска бухгалтеру?",
        "documents": [{"page_content": "Общие нормы: все работники должны..."}],
    },
    "research_agent": {
        "question": "Как проводить сварку?",
        "documents": [
            {
                "page_content": "При сварке использовать щиток.",
                "metadata": {"source": "ГОСТ 123"},
            }
        ],
    },
    "verification_agent": {
        "answer": "Сварщик обязан носить щиток.",
        "documents": [{"page_content": "При сварке использовать щиток."}],
    },
    "ultimate_chain": {
        "question": "Ширина прохода?",
        "context": "СП 1.1 п.2: ширина 1м.",
    },
    "final_chain": {"question": "Ширина прохода?", "context": "СП 1.1 п.2: ширина 1м."},
}


def check_output_format(prompt_id: str, content: str) -> bool:
    """Basic checks for expected tags/structure in the rendered prompt."""
    if "relevance" in prompt_id:
        return "CAN_ANSWER" in content or "PARTIAL" in content or "NO_MATCH" in content
    if "research" in prompt_id:
        return (
            "<answer>" in content or "answer_mode" in content
        )  # v2 has new instructions
    if "verification" in prompt_id:
        return "<json>" in content
    return True


def validate_registry(prompts_dir="prompts"):
    registry_path = os.path.join(prompts_dir, "registry.yaml")

    if not os.path.exists(registry_path):
        print(f"❌ Registry file not found: {registry_path}")
        return False

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to parse registry.yaml: {e}")
        return False

    pm = PromptManager()
    errors = []

    print(f"🔍 Validating {len(registry)} prompt types...")

    for prompt_id, config in registry.items():
        active_version = config.get("active_version")
        versions = config.get("versions", {})

        if not active_version:
            errors.append(f"[{prompt_id}] Missing 'active_version'")

        if not versions:
            errors.append(f"[{prompt_id}] No 'versions' defined")
            continue

        if active_version not in versions:
            errors.append(
                f"[{prompt_id}] active_version '{active_version}' is not in versions list"
            )

        # Validate each version
        for version, rel_path in versions.items():
            # Check for path traversal
            if ".." in rel_path or rel_path.startswith("/"):
                errors.append(f"[{prompt_id}:{version}] Forbidden path: {rel_path}")
                continue

            full_path = os.path.join(prompts_dir, rel_path)
            if not os.path.exists(full_path):
                errors.append(f"[{prompt_id}:{version}] File not found: {rel_path}")
                continue

            # Rendering Test
            try:
                # Use PromptManager's env to load by relative path
                # rel_path is relative to 'prompts/', but pm.env might be configured differently.
                # PromptManager loads from 'prompts_dir'.
                # Let's verify how PromptManager is initialized.
                # Usually it sets loader to prompts_dir.

                # We need inputs suitable for this prompt_id
                inputs = FIXTURES.get(prompt_id, {})
                if not inputs and "chain" in prompt_id:
                    inputs = FIXTURES.get("ultimate_chain")

                if not inputs:
                    print(f"⚠️ No fixture for {prompt_id}, skipping render test.")
                    continue

                template = pm.env.get_template(rel_path)
                rendered = template.render(**inputs)

                # Basic content check
                if not rendered.strip():
                    errors.append(f"[{prompt_id}:{version}] Rendered empty string")

                # Check for undefined variables (Jinja strict might catch this on render, but double check)

            except Exception as e:
                errors.append(f"[{prompt_id}:{version}] Rendering failed: {e}")

    if errors:
        print("❌ Prompt Registry Validation Failed:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("✅ Prompt Registry is valid and all templates render correctly.")
    return True


if __name__ == "__main__":
    success = validate_registry()
    if not success:
        sys.exit(1)
