# Third Party
from linkml_runtime.dumpers import YAMLDumper
from linkml_runtime.loaders import yaml_loader

# Local
from risk_atlas_nexus.ai_risk_ontology import Action, Container, Risk


SCHEMA_DIR = "src/ai_risk_ontology/schema/"
DATA_DIR = "src/data/knowledge-graph/"


def create_container_object() -> Container:

    # Add risk-related actions to risks
    actions = get_actions()
    risk_data = get_risks()
    new_risks = []
    for risk in risk_data.risks:
        risk.hasRelatedAction = [
            action.id
            for action in actions
            if action.hasRelatedRisk is not None and risk.id in action.hasRelatedRisk
        ]
        new_risks.append(risk)
    # Create container
    container = Container(
        risks=new_risks, documents=risk_data.documents, taxonomies=risk_data.taxonomies
    )
    return container


def get_risks() -> Container:
    instances = yaml_loader.load(DATA_DIR + "nist_ai_rmf_data.yaml", Container)
    return instances


def get_related_risks(related_risks_str: str, risks: dict[str, str]) -> list[str]:
    related_risks = related_risks_str.split(";")
    return [
        id
        for name, id in risks.items()
        for risk in related_risks
        if name == risk.strip()
    ]


def get_actions() -> list[Action]:
    instances = yaml_loader.load(DATA_DIR + "nist_ai_rmf_actions_data.yaml", Container)
    return instances.actions


container = create_container_object()
with open(DATA_DIR + "nist_ai_rmf_data.yaml", "+tw", encoding="utf-8") as output_file:
    print(YAMLDumper().dumps(container), file=output_file)
    output_file.close()
