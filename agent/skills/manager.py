"""
Skill manager for managing skill lifecycle and operations.
"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from common.log import logger
from agent.skills.types import Skill, SkillEntry, SkillSnapshot
from agent.skills.loader import SkillLoader
from agent.skills.formatter import format_skill_entries_for_prompt

SKILLS_CONFIG_FILE = "skills_config.json"


class SkillManager:
    """Manages skills for an agent."""

    def __init__(
        self,
        builtin_dir: Optional[str] = None,
        custom_dir: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the skill manager.

        :param builtin_dir: Built-in skills directory (project root ``skills/``)
        :param custom_dir: Custom skills directory (workspace ``skills/``)
        :param config: Configuration dictionary
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.builtin_dir = builtin_dir or os.path.join(project_root, 'skills')
        self.custom_dir = custom_dir or os.path.join(project_root, 'workspace', 'skills')
        self.config = config or {}
        self._skills_config_path = os.path.join(self.custom_dir, SKILLS_CONFIG_FILE)

        # skills_config: full skill metadata keyed by name
        # { "web-fetch": {"name": ..., "description": ..., "source": ..., "enabled": true}, ... }
        self.skills_config: Dict[str, dict] = {}

        self.loader = SkillLoader()
        self.skills: Dict[str, SkillEntry] = {}

        # Load skills on initialization
        self.refresh_skills()

    def refresh_skills(self):
        """Reload all skills from builtin and custom directories, then sync config."""
        self.skills = self.loader.load_all_skills(
            builtin_dir=self.builtin_dir,
            custom_dir=self.custom_dir,
        )
        self._sync_skills_config()
        logger.debug(f"SkillManager: Loaded {len(self.skills)} skills")

    # ------------------------------------------------------------------
    # skills_config.json management
    # ------------------------------------------------------------------
    def _load_skills_config(self) -> Dict[str, dict]:
        """Load skills_config.json from custom_dir. Returns empty dict if not found."""
        if not os.path.exists(self._skills_config_path):
            return {}
        try:
            with open(self._skills_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"[SkillManager] Failed to load {SKILLS_CONFIG_FILE}: {e}")
        return {}

    def _save_skills_config(self):
        """Persist skills_config to custom_dir/skills_config.json."""
        os.makedirs(self.custom_dir, exist_ok=True)
        try:
            with open(self._skills_config_path, "w", encoding="utf-8") as f:
                json.dump(self.skills_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[SkillManager] Failed to save {SKILLS_CONFIG_FILE}: {e}")

    def _sync_skills_config(self):
        """
        Merge directory-scanned skills with the persisted config file.

        - New skills: use metadata.default_enabled as initial enabled state.
        - Existing skills: preserve their persisted enabled state.
        - Skills that no longer exist on disk are removed.
        - name/description/source are always refreshed from the latest scan.
        """
        saved = self._load_skills_config()
        merged: Dict[str, dict] = {}

        for name, entry in self.skills.items():
            skill = entry.skill
            prev = saved.get(name, {})
            category = prev.get("category", "skill")

            if name in saved:
                enabled = prev.get("enabled", True)
            else:
                enabled = entry.metadata.default_enabled if entry.metadata else True

            entry_dict = {
                "name": name,
                "description": skill.description,
                "source": prev.get("source") or skill.source,
                "enabled": enabled,
                "category": category,
            }
            display_name = prev.get("display_name")
            if display_name:
                entry_dict["display_name"] = display_name
            merged[name] = entry_dict

        self.skills_config = merged
        self._save_skills_config()

    def is_skill_enabled(self, name: str) -> bool:
        """
        Check if a skill is enabled according to skills_config.

        :param name: skill name
        :return: True if enabled (default True if not in config)
        """
        entry = self.skills_config.get(name)
        if entry is None:
            return True
        return entry.get("enabled", True)

    # ------------------------------------------------------------------
    # create / update
    # ------------------------------------------------------------------
    def create_skill(self, name: str, description: str, content: str) -> dict:
        """
        Create a new custom skill by writing a SKILL.md file.

        :param name: skill name (also used as directory name)
        :param description: short description
        :param content: full markdown body (frontmatter will be prepended)
        :return: skill metadata dict
        :raises ValueError: if name is empty or skill already exists
        """
        name = name.strip()
        if not name:
            raise ValueError("skill name is required")

        # Check for name collisions across both builtin and custom
        if name in self.skills_config:
            raise ValueError(f"skill '{name}' already exists")

        skill_dir = os.path.join(self.custom_dir, name)
        if os.path.exists(skill_dir):
            raise ValueError(f"skill directory '{name}' already exists")

        os.makedirs(skill_dir, exist_ok=True)

        # Write SKILL.md with YAML frontmatter
        skill_md = "---\n"
        skill_md += f"name: {name}\n"
        skill_md += f"description: {description}\n"
        skill_md += "---\n\n"
        skill_md += content.strip() + "\n"

        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        with open(skill_md_path, "w", encoding="utf-8") as f:
            f.write(skill_md)

        # Reload skills so the new one is picked up
        self.refresh_skills()

        entry = self.skills_config.get(name, {})
        logger.info(f"[SkillManager] Created skill '{name}' at {skill_md_path}")
        return entry

    def update_skill(self, name: str, updates: dict) -> dict:
        """
        Update an existing custom skill's metadata and/or content.

        :param name: skill name
        :param updates: dict with optional keys: description, content, enabled
        :return: updated skill metadata dict
        :raises ValueError: if skill not found or is builtin
        """
        name = name.strip()
        if name not in self.skills_config:
            raise ValueError(f"skill '{name}' not found")

        entry = self.skills_config[name]

        # Only custom skills can be edited via this method
        source = entry.get("source") or ""
        if source == "builtin":
            raise ValueError(f"builtin skill '{name}' cannot be edited")

        # Update description in config
        if "description" in updates:
            new_desc = (updates["description"] or "").strip()
            entry["description"] = new_desc

        # Update enabled state in config
        if "enabled" in updates:
            entry["enabled"] = bool(updates["enabled"])

        has_content_update = "content" in updates and updates["content"] is not None
        has_desc_update = "description" in updates

        # Rewrite SKILL.md if description or content changed, so that
        # refresh_skills() picks up the new values from the file.
        if has_content_update or has_desc_update:
            skill_dir = os.path.join(self.custom_dir, name)
            skill_md_path = os.path.join(skill_dir, "SKILL.md")

            if has_content_update:
                new_body = (updates["content"] or "").strip()
            else:
                # Preserve existing body from the file
                existing_body = ""
                if os.path.exists(skill_md_path):
                    try:
                        with open(skill_md_path, "r", encoding="utf-8") as f:
                            raw = f.read()
                        # Extract body after frontmatter
                        parts = raw.split("---\n", 2)
                        if len(parts) >= 3:
                            existing_body = parts[2].strip()
                        elif len(parts) == 2 and parts[0] == "":
                            existing_body = parts[1].strip()
                        else:
                            existing_body = raw.strip()
                    except Exception:
                        existing_body = ""
                new_body = existing_body

            md_content = f"---\nname: {name}\n"
            md_content += f"description: {entry.get('description', '')}\n"
            md_content += "---\n\n"
            md_content += new_body + "\n"

            os.makedirs(skill_dir, exist_ok=True)
            with open(skill_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

        self._save_skills_config()

        # Refresh to re-read skill content
        self.refresh_skills()

        logger.info(f"[SkillManager] Updated skill '{name}'")
        return dict(self.skills_config.get(name, {}))

    def set_skill_enabled(self, name: str, enabled: bool):
        """
        Set a skill's enabled state and persist.

        :param name: skill name
        :param enabled: True to enable, False to disable
        """
        if name not in self.skills_config:
            raise ValueError(f"skill '{name}' not found in config")
        self.skills_config[name]["enabled"] = enabled
        self._save_skills_config()

    def get_skills_config(self) -> Dict[str, dict]:
        """
        Return the full skills_config dict (for query API).

        :return: copy of skills_config
        """
        return dict(self.skills_config)
    
    def get_skill(self, name: str) -> Optional[SkillEntry]:
        """
        Get a skill by name.
        
        :param name: Skill name
        :return: SkillEntry or None if not found
        """
        return self.skills.get(name)
    
    def list_skills(self) -> List[SkillEntry]:
        """
        Get all loaded skills.
        
        :return: List of all skill entries
        """
        return list(self.skills.values())
    
    @staticmethod
    def _normalize_skill_filter(skill_filter: Optional[List[str]]) -> Optional[List[str]]:
        """Normalize a skill_filter list into a flat list of stripped names."""
        if skill_filter is None:
            return None
        normalized = []
        for item in skill_filter:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    normalized.append(name)
            elif isinstance(item, list):
                for subitem in item:
                    if isinstance(subitem, str):
                        name = subitem.strip()
                        if name:
                            normalized.append(name)
        return normalized or None

    def filter_skills(
        self,
        skill_filter: Optional[List[str]] = None,
        include_disabled: bool = False,
    ) -> List[SkillEntry]:
        """
        Filter skills that are eligible (enabled + requirements met).

        :param skill_filter: List of skill names to include (None = all)
        :param include_disabled: Whether to include disabled skills
        :return: Filtered list of eligible skill entries
        """
        from agent.skills.config import should_include_skill

        entries = list(self.skills.values())

        entries = [e for e in entries if should_include_skill(e, self.config)]

        normalized = self._normalize_skill_filter(skill_filter)
        if normalized is not None:
            entries = [e for e in entries if e.skill.name in normalized]

        if not include_disabled:
            entries = [e for e in entries if self.is_skill_enabled(e.skill.name)]

        from config import conf
        if not conf().get("knowledge", True):
            entries = [e for e in entries if e.skill.name != "knowledge-wiki"]

        return entries

    def filter_unavailable_skills(
        self,
        skill_filter: Optional[List[str]] = None,
    ) -> tuple:
        """
        Find skills that are enabled but have unmet requirements.

        :param skill_filter: Optional list of skill names to include
        :return: Tuple of (entries, missing_map) where missing_map maps
                 skill name to its missing requirements dict
        """
        from agent.skills.config import should_include_skill, get_missing_requirements

        entries = list(self.skills.values())

        # Only enabled skills
        entries = [e for e in entries if self.is_skill_enabled(e.skill.name)]

        normalized = self._normalize_skill_filter(skill_filter)
        if normalized is not None:
            entries = [e for e in entries if e.skill.name in normalized]

        # Keep only those that fail should_include_skill (requirements not met)
        unavailable = []
        missing_map: Dict[str, dict] = {}
        for e in entries:
            if not should_include_skill(e, self.config):
                missing = get_missing_requirements(e)
                if missing:
                    unavailable.append(e)
                    missing_map[e.skill.name] = missing

        return unavailable, missing_map

    def build_skills_prompt(
        self,
        skill_filter: Optional[List[str]] = None,
    ) -> str:
        """
        Build a formatted prompt containing available skills
        and brief hints for unavailable ones.

        :param skill_filter: Optional list of skill names to include
        :return: Formatted skills prompt
        """
        from common.log import logger
        from agent.skills.formatter import format_unavailable_skills_for_prompt

        eligible = self.filter_skills(skill_filter=skill_filter, include_disabled=False)
        logger.debug(f"[SkillManager] Eligible: {len(eligible)} skills (total: {len(self.skills)})")
        if eligible:
            skill_names = [e.skill.name for e in eligible]
            logger.debug(f"[SkillManager] Eligible skills: {skill_names}")

        result = format_skill_entries_for_prompt(eligible)

        unavailable, missing_map = self.filter_unavailable_skills(skill_filter=skill_filter)
        if unavailable:
            unavailable_names = [e.skill.name for e in unavailable]
            logger.debug(f"[SkillManager] Unavailable skills (setup needed): {unavailable_names}")
            result += format_unavailable_skills_for_prompt(unavailable, missing_map)

        logger.debug(f"[SkillManager] Generated prompt length: {len(result)}")
        return result
    
    def build_skill_snapshot(
        self,
        skill_filter: Optional[List[str]] = None,
        version: Optional[int] = None,
    ) -> SkillSnapshot:
        """
        Build a snapshot of skills for a specific run.
        
        :param skill_filter: Optional list of skill names to include
        :param version: Optional version number for the snapshot
        :return: SkillSnapshot
        """
        entries = self.filter_skills(skill_filter=skill_filter, include_disabled=False)
        prompt = format_skill_entries_for_prompt(entries)
        
        skills_info = []
        resolved_skills = []
        
        for entry in entries:
            skills_info.append({
                'name': entry.skill.name,
                'primary_env': entry.metadata.primary_env if entry.metadata else None,
            })
            resolved_skills.append(entry.skill)
        
        return SkillSnapshot(
            prompt=prompt,
            skills=skills_info,
            resolved_skills=resolved_skills,
            version=version,
        )
    
    def sync_skills_to_workspace(self, target_workspace_dir: str):
        """
        Sync all loaded skills to a target workspace directory.
        
        This is useful for sandbox environments where skills need to be copied.
        
        :param target_workspace_dir: Target workspace directory
        """
        import shutil
        
        target_skills_dir = os.path.join(target_workspace_dir, 'skills')
        
        # Remove existing skills directory
        if os.path.exists(target_skills_dir):
            shutil.rmtree(target_skills_dir)
        
        # Create new skills directory
        os.makedirs(target_skills_dir, exist_ok=True)
        
        # Copy each skill
        for entry in self.skills.values():
            skill_name = entry.skill.name
            source_dir = entry.skill.base_dir
            target_dir = os.path.join(target_skills_dir, skill_name)
            
            try:
                shutil.copytree(source_dir, target_dir)
                logger.debug(f"Synced skill '{skill_name}' to {target_dir}")
            except Exception as e:
                logger.warning(f"Failed to sync skill '{skill_name}': {e}")
        
        logger.info(f"Synced {len(self.skills)} skills to {target_skills_dir}")
    
    def get_skill_by_key(self, skill_key: str) -> Optional[SkillEntry]:
        """
        Get a skill by its skill key (which may differ from name).
        
        :param skill_key: Skill key to look up
        :return: SkillEntry or None
        """
        for entry in self.skills.values():
            if entry.metadata and entry.metadata.skill_key == skill_key:
                return entry
            if entry.skill.name == skill_key:
                return entry
        return None
