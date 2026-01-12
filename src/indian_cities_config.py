"""
Indian Cities Configuration
Centralized metadata for 30+ major Indian cities including coordinates,
demographics, pollution characteristics, and socioeconomic data for equity analysis.

Data is loaded from the external JSON file: data/indian_cities.json
This allows easier updates, validation, and data provenance tracking.
"""

import json
import os
from pathlib import Path

# Determine paths correctly whether running from src/ or project root
_current_file = Path(__file__)
_src_dir = _current_file.parent
_project_root = _src_dir.parent

# Try multiple paths for the JSON file
_json_paths = [
    _project_root / "data" / "indian_cities.json",  # When running from project root
    _src_dir.parent / "data" / "indian_cities.json",  # Alternative path
    Path("data/indian_cities.json"),  # Relative to CWD
]

_config_data = None

def _load_config():
    """Load configuration from JSON file (cached)."""
    global _config_data
    if _config_data is not None:
        return _config_data
    
    for json_path in _json_paths:
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                _config_data = json.load(f)
            return _config_data
    
    raise FileNotFoundError(
        f"Could not find indian_cities.json. Searched: {[str(p) for p in _json_paths]}"
    )

def _get_config():
    """Get the loaded configuration data."""
    return _load_config()


# Lazy-loaded data accessors (maintain backward compatibility)
def _get_cities():
    return _get_config().get("cities", {})

def _get_policies():
    return _get_config().get("policies", {})

def _get_aqi_categories():
    return _get_config().get("aqi_categories", {})


# Create module-level variables for backward compatibility
# These are properties that load data on first access
class _LazyDict(dict):
    """A dict that loads data on first access."""
    def __init__(self, loader_func):
        super().__init__()
        self._loader_func = loader_func
        self._loaded = False
    
    def _ensure_loaded(self):
        if not self._loaded:
            self.update(self._loader_func())
            self._loaded = True
    
    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)
    
    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()
    
    def __len__(self):
        self._ensure_loaded()
        return super().__len__()
    
    def keys(self):
        self._ensure_loaded()
        return super().keys()
    
    def values(self):
        self._ensure_loaded()
        return super().values()
    
    def items(self):
        self._ensure_loaded()
        return super().items()
    
    def get(self, key, default=None):
        self._ensure_loaded()
        return super().get(key, default)


# Backward-compatible module-level exports
INDIAN_CITIES = _LazyDict(_get_cities)
POLICY_INTERVENTIONS = _LazyDict(_get_policies)
AQI_CATEGORIES = _LazyDict(_get_aqi_categories)


# Helper functions
def get_city_metadata(city_name):
    """Get metadata for a specific city."""
    return INDIAN_CITIES.get(city_name, None)


def get_all_cities():
    """Get list of all configured cities."""
    return list(INDIAN_CITIES.keys())


def get_cities_by_tier(tier):
    """Get cities by tier (1, 2, or 3)."""
    return [city for city, data in INDIAN_CITIES.items() if data.get("tier") == tier]


def get_cities_by_pollution_level(threshold_aqi=200):
    """Get cities with baseline winter AQI above threshold."""
    return [
        city for city, data in INDIAN_CITIES.items() 
        if data.get("baseline_winter_aqi", 0) >= threshold_aqi
    ]


def get_aqi_category(aqi_value):
    """Get AQI category and color based on value."""
    for category, info in AQI_CATEGORIES.items():
        range_vals = info.get("range", [0, 0])
        min_val, max_val = range_vals[0], range_vals[1]
        if min_val <= aqi_value <= max_val:
            return category, info.get("color", "#888888"), info.get("health_impact", "Unknown")
    return "Severe", "#7e0023", "Hazardous"


def get_config_metadata():
    """Get metadata about the configuration (version, last updated, source)."""
    return _get_config().get("metadata", {})


if __name__ == "__main__":
    print("=" * 60)
    print("  INDIAN CITIES CONFIGURATION (JSON-Loaded)")
    print("=" * 60)
    
    metadata = get_config_metadata()
    print(f"\nData Source: {metadata.get('source', 'Unknown')}")
    print(f"Last Updated: {metadata.get('last_updated', 'Unknown')}")
    print(f"Version: {metadata.get('version', 'Unknown')}")
    
    print(f"\nTotal cities configured: {len(INDIAN_CITIES)}")
    print(f"Tier-1 cities: {len(get_cities_by_tier(1))}")
    print(f"Tier-2 cities: {len(get_cities_by_tier(2))}")
    print(f"\nHighly polluted cities (Winter AQI > 200): {len(get_cities_by_pollution_level(200))}")
    print(get_cities_by_pollution_level(200))
    
    print(f"\nPolicies defined: {list(POLICY_INTERVENTIONS.keys())}")
    print(f"AQI categories: {list(AQI_CATEGORIES.keys())}")
