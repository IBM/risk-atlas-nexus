# Convenience script to export a tex version or IBM AI risk atlas from the yaml
# in src/risk_atlas_nexus/data/knowledge_graph

import os

from risk_atlas_nexus import RiskAtlasNexus
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import (
    Container,
    Documentation,
    RiskGroup,
    RiskTaxonomy,
)
from risk_atlas_nexus.ai_risk_ontology.util.latex_dumper import LatexDumper
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)

ran = RiskAtlasNexus()


def create_container_object() -> Container:
    # Create paper
    documents = [
        Documentation(
            **{
                "id": "10a99803d8afd656",
                "name": "Foundation models: Opportunities, risks and mitigations",
                "description": "In this document we: Explore the benefits of foundation models, including their "
                "capability to perform challenging tasks, potential to speed up the adoption of AI, "
                "ability to increase productivity and the cost benefits they provide. Discuss the three "
                "categories of risk, including risks known from earlier forms of AI, known risks "
                "amplified by foundation models and emerging risks intrinsic to the generative "
                "capabilities of foundation models. Cover the principles, pillars and governance that "
                "form the foundation of IBM’s AI ethics initiatives and suggest guardrails for risk "
                "mitigation.",
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
                "description": "Explore this atlas to understand some of the risks of working with generative AI, "
                "foundation models, and machine learning models.",
                "url": "https://www.ibm.com/docs/en/watsonx/saas?topic=ai-risk-atlas",
                "hasDocumentation": ["10a99803d8afd656"],
            }
        )
    ]

    # Get risks
    risks = ran.get_all_risks("ibm-risk-atlas")

    # Create container
    container = Container(documents=documents, taxonomies=taxonomies, risks=risks)
    return container


# export IBM AI risk atlas to latex
export_path = "graph_export/latex"
if not os.path.isdir(export_path):
    logger.error(f"Directory %s does not exist.", export_path)
    raise FileNotFoundError("Export directory is not found", export_path)

export_file_path = os.path.join(export_path, "ibm-ai-risk-atlas-risks.tex")
container = create_container_object()

with open(export_file_path, "+tw", encoding="utf-8") as output_file:
    print(LatexDumper().dumps(container), file=output_file)
    output_file.close()
