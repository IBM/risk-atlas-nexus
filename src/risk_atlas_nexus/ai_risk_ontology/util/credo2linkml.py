# Standard Library
import ast
import csv

# Third Party
from linkml_runtime.dumpers import YAMLDumper

# Local
from risk_atlas_nexus.ai_risk_ontology import (
    Action,
    Container,
    Documentation,
    Risk,
    RiskControl,
    RiskGroup,
    RiskTaxonomy,
)


def create_container_object() -> Container:
    # Create risk taxonomy paper from Credo
    documents = [
        Documentation(
            **{
                "id": "credo-doc",
                "name": "The Unified Control Framework: Establishing a Common Foundation for Enterprise AI Governance, Risk Management and Regulatory Compliance",
                "description": "The rapid advancement and deployment of AI systems have created an urgent need for standard safety-evaluation frameworks. This paper introduces AILuminate v1.0, the first comprehensive industry-standard benchmark for assessing AI-product risk and reliability. Its development employed an open process that included participants from multiple fields. The benchmark evaluates an AI system's resistance to prompts designed to elicit dangerous, illegal, or undesirable behavior in 12 hazard categories, including violent crimes, nonviolent crimes, sex-related crimes, child sexual exploitation, indiscriminate weapons, suicide and self-harm, intellectual property, privacy, defamation, hate, sexual content, and specialized advice (election, financial, health, legal). Our method incorporates a complete assessment standard, extensive prompt datasets, a novel evaluation framework, a grading and reporting system, and the technical as well as organizational infrastructure for long-term support and evolution. In particular, the benchmark employs an understandable five-tier grading scale (Poor to Excellent) and incorporates an innovative entropy-based system-response evaluation. In addition to unveiling the benchmark, this report also identifies limitations of our method and of building safety benchmarks generally, including evaluator uncertainty and the constraints of single-turn interactions. This work represents a crucial step toward establishing global standards for AI risk and reliability evaluation while acknowledging the need for continued development in areas such as multiturn interactions, multimodal understanding, coverage of additional languages, and emerging hazard categories. Our findings provide valuable insights for model developers, system integrators, and policymakers working to promote safer AI deployment.",
                "url": "https://arxiv.org/pdf/2503.05937v1",
                "dateCreated": "2025-03-07",
                "dateModified": "2025-03-07",
            }
        )
    ]

    # Create risk taxonomy
    taxonomies = [
        RiskTaxonomy(
            **{
                "id": "credo-ucf",
                "name": "Credo Unified Control Framework",
                "description": "A comprehensive risk taxonomy synthesizing organizational and societal risks",
                "url": "https://arxiv.org/abs/2503.05937v1",
                "version": "1.0",
                "dateCreated": "2025-03-07",
                "hasDocumentation": ["credo-doc"],
            }
        )
    ]

    # Create risk groups
    risks = get_risks()
    risk_group_names = sorted(list(set([risk["Risk Type"] for risk in risks])))
    riskgroups = [
        RiskGroup(
            **{
                "id": "credo-rg-"
                + risk_group.lower().replace(" ", "-").replace("&", "and"),
                "name": risk_group,
                "isDefinedByTaxonomy": "credo-ucf",
            }
        )
        for risk_group in risk_group_names
    ]

    # Create risks
    risk_objects = [
        Risk(
            **{
                "id": "credo-" + risk["RISK ID"].lower().replace(" ", "-"),
                "name": risk["Risk Scenario"],
                "description": risk["Description"],
                "isPartOf": "credo-rg-" + risk["Risk Type"].lower().replace(" ", "-"),
                "hasRelatedAction": [
                    "credo-act-" + c.lower().replace(" ", "-")
                    for c in ast.literal_eval(risk["Control ID"])
                ],
                "isDefinedByTaxonomy": "credo-ucf",
            }
        )
        for risk in risks
    ]

    # Create risk actions
    actions = get_risk_actions()

    acts = [
        Action(
            **{
                "id": "credo-act-" + action["Control ID"].lower().replace(" ", "-"),
                "name": action["Control Label"],
                "description": action["Description"],
                "dateCreated": "2025-03-07",
                "dateModified": "2025-03-07",
                "hasDocumentation": ["credo-doc"],
                "hasRelatedRisk": [
                    "credo-" + r.lower().replace(" ", "-")
                    for r in ast.literal_eval(action["RISK ID"])
                ],
                "isDefinedByTaxonomy": "credo-ucf",
            }
        )
        for action in actions
    ]

    # Create container
    container = Container(
        documents=documents,
        taxonomies=taxonomies,
        riskgroups=riskgroups,
        risks=risk_objects,
        actions=acts,
    )
    return container


def get_risks() -> list[dict]:
    with open(
        "resources/credo-risks-with-ctl.csv",
        encoding="utf-8",
    ) as csvfile:
        importer = csv.DictReader(csvfile)
        return [row for row in importer]


def get_risk_actions() -> list[dict]:
    with open(
        "resources/credo-ctl-with-risk.csv",
        encoding="utf-8",
    ) as csvfile:
        importer = csv.DictReader(csvfile)
        return [row for row in importer]


with open(
    "src/risk_atlas_nexus/data/knowledge_graph/credo.yaml",
    "+tw",
    encoding="utf-8",
) as output_file:
    container = create_container_object()
    print(YAMLDumper().dumps(container), file=output_file)
    output_file.close()
