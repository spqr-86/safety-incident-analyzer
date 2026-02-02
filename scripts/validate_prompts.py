import os
import yaml
import sys


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

    errors = []

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

        for version, rel_path in versions.items():
            # Check for path traversal
            if ".." in rel_path or rel_path.startswith("/"):
                errors.append(f"[{prompt_id}:{version}] Forbidden path: {rel_path}")
                continue

            full_path = os.path.join(prompts_dir, rel_path)
            if not os.path.exists(full_path):
                errors.append(f"[{prompt_id}:{version}] File not found: {rel_path}")

    if errors:
        print("❌ Prompt Registry Validation Failed:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("✅ Prompt Registry is valid.")
    return True


if __name__ == "__main__":
    success = validate_registry()
    if not success:
        sys.exit(1)
