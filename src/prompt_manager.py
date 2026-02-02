import os
import yaml
import hashlib
import logging
from typing import Any, Dict, Optional
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound


class PromptManager:
    def __init__(
        self, prompts_dir: str = "prompts", registry_file: str = "registry.yaml"
    ):
        self.prompts_dir = os.path.abspath(prompts_dir)
        self.registry_path = os.path.join(self.prompts_dir, registry_file)
        self.env = Environment(
            loader=FileSystemLoader(self.prompts_dir),
            undefined=StrictUndefined,
            autoescape=False,
        )
        self.registry = self._load_registry()
        self.logger = logging.getLogger("PromptManager")

    def _load_registry(self) -> Dict[str, Any]:
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(f"Registry file not found at {self.registry_path}")
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_version(self, prompt_id: str) -> str:
        # Приоритет 1: ENV variable (PROMPT_ID_VERSION)
        env_key = f"PROMPT_{prompt_id.upper()}_VERSION"
        if env_key in os.environ:
            version = os.environ[env_key]
            self.logger.debug(f"Resolved {prompt_id} version from ENV: {version}")
            return version

        # Приоритет 2: Registry active_version
        if prompt_id in self.registry:
            version = self.registry[prompt_id].get("active_version")
            if version:
                return version

        raise ValueError(f"Could not resolve version for prompt_id: {prompt_id}")

    def _get_template_path(self, prompt_id: str, version: str) -> str:
        if prompt_id not in self.registry:
            raise KeyError(f"Prompt ID '{prompt_id}' not found in registry")

        versions = self.registry[prompt_id].get("versions", {})
        if version not in versions:
            raise KeyError(
                f"Version '{version}' for prompt '{prompt_id}' not found in registry"
            )

        rel_path = versions[version]

        # Безопасность: запрет на traversal
        if ".." in rel_path or rel_path.startswith("/"):
            raise ValueError(
                f"Invalid template path: {rel_path}. Absolute paths and '..' are forbidden."
            )

        return rel_path

    def render(self, prompt_id: str, **kwargs) -> str:
        version = self._resolve_version(prompt_id)
        template_path = self._get_template_path(prompt_id, version)

        try:
            template = self.env.get_template(template_path)
            rendered = template.render(**kwargs)

            # Хеширование и логирование
            prompt_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            self.logger.info(
                f"Rendered prompt | ID: {prompt_id} | Version: {version} | "
                f"Hash: {prompt_hash[:8]} | Inputs: {list(kwargs.keys())}"
            )

            # Полный текст только в DEBUG
            if os.environ.get("DEBUG_PROMPTS") == "true":
                self.logger.debug(f"Full prompt [{prompt_id}:{version}]:\n{rendered}")

            return rendered

        except TemplateNotFound:
            raise FileNotFoundError(
                f"Template file '{template_path}' not found in {self.prompts_dir}"
            )
        except Exception as e:
            self.logger.error(f"Error rendering prompt '{prompt_id}': {str(e)}")
            raise
