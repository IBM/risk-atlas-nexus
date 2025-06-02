# Standard Library
import csv

# Third Party
from linkml_runtime.dumpers import YAMLDumper

# Local
from risk_atlas_nexus.ai_risk_ontology import (
    Container,
    Documentation,
    Risk,
    RiskGroup,
    RiskTaxonomy,
)


def create_container_object() -> Container:
    # Create risk taxonomy paper
    documents = [
        Documentation(
            **{
                "id": "arxiv.org/2408.12622",
                "name": "The AI Risk Repository: A Comprehensive Meta-Review, Database, and Taxonomy of Risks From Artificial Intelligence",
                "description": "The risks posed by Artificial Intelligence (AI) are of considerable concern to academics, auditors, policymakers, AI companies, and the public. However, a lack of shared understanding of AI risks can impede our ability to comprehensively discuss, research, and react to them. This paper addresses this gap by creating an AI Risk Repository to serve as a common frame of reference. This comprises a living database of 777 risks extracted from 43 taxonomies, which can be filtered based on two overarching taxonomies and easily accessed, modified, and updated via our website and online spreadsheets. We construct our Repository with a systematic review of taxonomies and other structured classifications of AI risk followed by an expert consultation. We develop our taxonomies of AI risk using a best-fit framework synthesis. Our high-level Causal Taxonomy of AI Risks classifies each risk by its causal factors (1) Entity: Human, AI; (2) Intentionality: Intentional, Unintentional; and (3) Timing: Pre-deployment; Post-deployment. Our mid-level Domain Taxonomy of AI Risks classifies risks into seven AI risk domains: (1) Discrimination & toxicity, (2) Privacy & security, (3) Misinformation, (4) Malicious actors & misuse, (5) Human-computer interaction, (6) Socioeconomic & environmental, and (7) AI system safety, failures, & limitations. These are further divided into 23 subdomains. The AI Risk Repository is, to our knowledge, the first attempt to rigorously curate, analyze, and extract AI risk frameworks into a publicly accessible, comprehensive, extensible, and categorized risk database. This creates a foundation for a more coordinated, coherent, and complete approach to defining, auditing, and managing the risks posed by AI systems.",
                "url": "https://arxiv.org/abs/2408.12622",
                "dateCreated": "2024-08-14",
                "dateModified": "2024-08-14",
            }
        )
    ]

    # Create risk taxonomy
    taxonomies = [
        RiskTaxonomy(
            **{
                "id": "mit-ai-risk-repository",
                "name": "The AI Risk Repository",
                "description": "A comprehensive living database of over 700 AI risks categorized by their cause and risk domain.",
                "url": "https://airisk.mit.edu/",
                "version": "1",
                "dateCreated": "2024-08-16",
                "hasDocumentation": ["arxiv.org/2408.12622"],
            }
        )
    ]

    # Create risk groups
    risks = get_risks()
    risk_group_names = sorted(
        list(set([(risk["Major"], risk["Domain"]) for risk in risks]))
    )
    riskgroups = [
        RiskGroup(
            **{
                "id": "mit-ai-risk-domain-" + risk_group[0],
                "name": risk_group[1],
                "isDefinedByTaxonomy": "mit-ai-risk-repository",
            }
        )
        for risk_group in risk_group_names
    ]

    # Create risks
    risk_objects = [
        Risk(
            **{
                "id": "mit-ai-risk-subdomain-" + risk["Minor"],
                "name": risk["Sub-domain"],
                "description": risk["Description"],
                "isPartOf": "mit-ai-risk-domain-" + risk["Major"],
                "isDefinedByTaxonomy": "mit-ai-risk-repository",
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
    with open("resources/TheAIRiskRepositoryV1_16_8_24.csv") as csvfile:
        importer = csv.DictReader(csvfile)
        return [row for row in importer]


with open(
    "src/risk_atlas_nexus/data/knowledge_graph/mit_ai_risk_repository_data.yaml",
    "+tw",
    encoding="utf-8",
) as output_file:
    container = create_container_object()
    print(YAMLDumper().dumps(container), file=output_file)
    output_file.close()
