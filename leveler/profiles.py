import json
import os
from dataclasses import asdict
from PyQt6.QtCore import QStandardPaths
from .analysis import Settings

class ProfileManager:
    def __init__(self):
        self.config_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation)
        if not self.config_dir:
            # Fallback to current directory if AppConfigLocation is not available
            self.config_dir = os.path.join(os.getcwd(), "config")
        
        os.makedirs(self.config_dir, exist_ok=True)
        self.profiles_path = os.path.join(self.config_dir, "profiles.json")
        self.profiles = self.load_profiles()

    def load_profiles(self) -> dict:
        if not os.path.exists(self.profiles_path):
            return {}
        try:
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_profiles(self):
        try:
            with open(self.profiles_path, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, indent=4)
        except Exception:
            pass

    def get_profile_names(self) -> list[str]:
        return sorted(self.profiles.keys())

    def get_settings(self, name: str) -> Settings | None:
        if name not in self.profiles:
            return None
        data = self.profiles[name]
        return Settings(**data)

    def save_profile(self, name: str, settings: Settings):
        self.profiles[name] = asdict(settings)
        self.save_profiles()

    def delete_profile(self, name: str):
        if name in self.profiles:
            del self.profiles[name]
            self.save_profiles()
