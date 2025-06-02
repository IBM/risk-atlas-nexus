from pathlib import Path

from .ai_risk_ontology import *


THIS_PATH = Path(__file__).parent

SCHEMA_DIRECTORY = THIS_PATH.parent / "schema"
MAIN_SCHEMA_PATH = SCHEMA_DIRECTORY / "ai_risk_ontology.yaml"
