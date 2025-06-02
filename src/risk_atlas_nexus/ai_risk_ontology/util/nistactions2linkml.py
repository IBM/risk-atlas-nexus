# Standard Library
import csv

# Third Party
from linkml_runtime.dumpers import YAMLDumper
from linkml_runtime.loaders import yaml_loader

# Local
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import *


SCHEMA_DIR = "src/ai_risk_ontology/schema/"
DATA_DIR = "src/data/knowledge-graph/"


def create_container_object() -> Container:

    # Create risk-related actions
    action_list = get_actions()
    risks = get_risks()
    actions = [
        Action(
            id=action["NIST Action ID"],
            name=action["NIST Action ID"],
            description=action["NIST Title"],
            hasAiActorTask=[
                task.strip() for task in action["AI Actor Tasks"].split(",")
            ],
            hasRelatedRisk=get_related_risks(action["NIST GAI Risks"], risks),
            hasDocumentation=["NIST.AI.600-1"],
        )
        for action in action_list
    ]
    # Create container
    container = Container(actions=actions)
    return container


def get_risks() -> dict[str, str]:
    instances = yaml_loader.load(DATA_DIR + "nist_ai_rmf_data.yaml", Container)
    return {risk.name: risk.id for risk in instances.risks}


def get_related_risks(related_risks_str: str, risks: dict[str, str]) -> list[str]:
    related_risks = related_risks_str.split(";")
    return [
        id
        for name, id in risks.items()
        for risk in related_risks
        if name == risk.strip()
    ]


def get_actions() -> list[dict]:
    with open("resources/actions_extracted_from_nist.csv") as csvfile:
        importer = csv.DictReader(csvfile)
        return [row for row in importer]


with open(
    DATA_DIR + "nist_ai_rmf_actions_data.yaml", "+tw", encoding="utf-8"
) as output_file:
    container = create_container_object()
    print(YAMLDumper().dumps(container), file=output_file)
    output_file.close()
