# Standard Library
from datetime import datetime

# Third Party
from linkml_runtime.dumpers import YAMLDumper
from requests import get

# Local
from risk_atlas_nexus.ai_risk_ontology import (
    Container,
    Documentation,
    Risk,
    RiskGroup,
    RiskTaxonomy,
)


MO_HOST = "https://usage-gov-advisor.bx.cloud9.ibm.com"
MO_API_PREFIX = "/v1"
DATA_DIR = "src/risk_atlas_nexus/data/knowledge_graph/"


def convert_datetime_to_date(datetime_str: str) -> str:
    datetime_value = datetime.fromisoformat(datetime_str)
    return datetime_value.date().isoformat()


def create_container_object() -> Container:
    # Get risks from AirTable
    risks = get_risks()
    min_date = min(datetime.fromisoformat(risk["creation_date"]) for risk in risks)
    max_date = max(datetime.fromisoformat(risk["last_update_date"]) for risk in risks)

    # Create paper
    documents = [
        Documentation(
            **{
                "id": "10a99803d8afd656",
                "name": "Foundation models: Opportunities, risks and mitigations",
                "description": "In this document we: Explore the benefits of foundation models, including their capability to perform challenging tasks, potential to speed up the adoption of AI, ability to increase productivity and the cost benefits they provide. Discuss the three categories of risk, including risks known from earlier forms of AI, known risks amplified by foundation models and emerging risks intrinsic to the generative capabilities of foundation models. Cover the principles, pillars and governance that form the foundation of IBM’s AI ethics initiatives and suggest guardrails for risk mitigation.",
                "url": "https://www.ibm.com/downloads/documents/us-en/10a99803d8afd656",
            }
        )
    ]

    # Create taxonomy
    taxonomies = [
        RiskTaxonomy(
            **{
                "id": "ibm-risk-atlas",
                "name": "IBM AI Risk Atlas",
                "description": "Explore this atlas to understand some of the risks of working with generative AI, foundation models, and machine learning models.",
                "url": "https://www.ibm.com/docs/en/watsonx/saas?topic=ai-risk-atlas",
                "hasDocumentation": ["10a99803d8afd656"],
                "dateCreated": min_date.date().isoformat(),
                "dateModified": max_date.date().isoformat(),
            }
        )
    ]

    # Create risk groups
    risk_group_names = sorted(list(set([risk["group"] for risk in risks])))
    riskgroups = [
        RiskGroup(
            **{
                "id": "ibm-risk-atlas-" + risk_group.replace(" ", "-"),
                "name": risk_group.capitalize(),
                "isDefinedByTaxonomy": "ibm-risk-atlas",
            }
        )
        for risk_group in risk_group_names
    ]

    # Create risks
    risk_objects = [
        Risk(
            **{
                "id": "atlas-" + risk["tag"],
                "name": risk["title"],
                "tag": risk["tag"],
                "type": risk["type"],
                "phase": risk["phase"],
                "descriptor": risk["descriptor"],
                "description": risk["description"],
                "concern": risk["concern"],
                "url": "https://www.ibm.com/docs/en/watsonx/saas?topic=SSYOK8/wsj/ai-risk-atlas/"
                + risk["tag"]
                + ".html",
                "dateCreated": convert_datetime_to_date(risk["creation_date"]),
                "dateModified": convert_datetime_to_date(risk["last_update_date"]),
                "isPartOf": "ibm-risk-atlas-" + risk["group"].replace(" ", "-"),
                "isDefinedByTaxonomy": "ibm-risk-atlas",
            }
        )
        for risk in risks
    ]

    # Create container
    container = Container(
        documents=documents,
        taxonomies=taxonomies,
        riskgroups=riskgroups,
        risks=risk_objects,
    )
    return container


def get_risks() -> list[dict]:
    url = MO_HOST + MO_API_PREFIX + "/risks"
    response = get(url=url, headers={"Accept": "application/json"}, timeout=60)
    response.encoding = "UTF-8"
    return response.json()


if __name__ == "__main__":
    with open(
        DATA_DIR + "risk_atlas_data.yaml",
        "+tw",
        encoding="utf-8",
    ) as output_file:
        container = create_container_object()
        print(YAMLDumper().dumps(container), file=output_file)
        output_file.close()
