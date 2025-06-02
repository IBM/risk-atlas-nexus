# Standard Library
import os
import re

# Third Party
import pandas as pd
from datasets import load_dataset
from linkml_runtime.dumpers import YAMLDumper

# Local
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import *


SCHEMA_DIR = "src/ai_risk_ontology/schema/"
DATA_DIR = "src/risk_atlas_nexus/data/knowledge-graph/"
ds = load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")


def create_container_object() -> Container:
    # Create risk taxonomy paper
    documents = [
        Documentation(
            **{
                "id": "arxiv.org/pdf/2406.17864",
                "name": "The AI Risk Taxonomy (AIR 2024)",
                "description": "We present a comprehensive AI risk taxonomy derived from eight government policies from the European Union, United States, and China and 16 company policies worldwide, making a significant step towards establishing a unified language for generative AI safety evaluation. We identify 314 unique risk categories, organized into a four-tiered taxonomy. At the highest level, this taxonomy encompasses System & Operational Risks, Content Safety Risks, Societal Risks, and Legal & Rights Risks. The taxonomy establishes connections between various descriptions and approaches to risk, highlighting the overlaps and discrepancies between public and private sector conceptions of risk. By providing this unified framework, we aim to advance AI safety through information sharing across sectors and the promotion of best practices in risk mitigation for generative AI models and systems.",
                "url": "https://arxiv.org/pdf/2406.17864",
                "dateCreated": "2024-09-05",
                "dateModified": "2024-09-05",
            }
        )
    ]

    # Create risk taxonomy
    taxonomies = [
        RiskTaxonomy(
            **{
                "id": "ai-risk-taxonomy",
                "name": "The AI Risk Taxonomy (AIR 2024)",
                "description": "An AI risk taxonomy derived from eight government policies from the European Union, United States, and China and 16 company policies worldwide. It identifies 314 unique risk categories organized into a four-tiered taxonomy. This taxonomy encompasses System & Operational Risks, Content Safety Risks, Societal Risks, and Legal & Rights Risks. The taxonomy establishes connections between various descriptions and approaches to risk, highlighting the overlaps and discrepancies between public and private sector conceptions of risk. ",
                "url": "https://arxiv.org/pdf/2406.17864",
                "version": "1",
                "dateCreated": "2024-09-05",
                "hasDocumentation": ["arxiv.org/pdf/2406.17864"],
            }
        )
    ]

    # Create risk groups
    risk_groups = []
    risk_objects = []

    risks = get_risks()
    df = pd.DataFrame(
        columns=["cate-idx", "l2-name", "l3-name", "l4-name", "definition"]
    )
    df = pd.concat([df, pd.DataFrame(risks)], ignore_index=True)

    l2_grouping = df.groupby("l2-name", as_index=False)[["l3-name"]].agg(list)

    for i, group_l2 in l2_grouping.iterrows():
        risk_groups.append(
            RiskGroup(
                **{
                    "id": "ai-risk-taxonomy-"
                    + group_l2["l2-name"].lower().replace(" ", "-"),
                    "name": group_l2["l2-name"],
                    "isDefinedByTaxonomy": "ai-risk-taxonomy",
                    "narrowMatch": list(
                        set(
                            [
                                "ai-risk-taxonomy-" + cat.lower().replace(" ", "-")
                                for cat in group_l2["l3-name"]
                            ]
                        )
                    ),
                }
            )
        )

    l3_grouping = df.groupby("l3-name", as_index=False)[["l2-name"]].agg(list)
    for i, group_l3 in l3_grouping.iterrows():
        risk_groups.append(
            RiskGroup(
                **{
                    "id": "ai-risk-taxonomy-"
                    + group_l3["l3-name"].lower().replace(" ", "-"),
                    "name": group_l3["l3-name"],
                    "isDefinedByTaxonomy": "ai-risk-taxonomy",
                    "broadMatch": list(
                        set(
                            [
                                "ai-risk-taxonomy-" + cat.lower().replace(" ", "-")
                                for cat in group_l3["l2-name"]
                            ]
                        )
                    ),
                }
            )
        )

    for risk in risks:
        risk_objects.append(
            Risk(
                **{
                    "id": "ai-risk-taxonomy-"
                    + risk["l4-name"].lower().replace(" ", "-"),
                    "name": risk["l4-name"],
                    "description": risk["definition"],
                    "isPartOf": "ai-risk-taxonomy-"
                    + risk["l3-name"].lower().replace(" ", "-"),
                    "isDefinedByTaxonomy": "ai-risk-taxonomy",
                }
            )
        )

    # Create container
    container = Container(
        documents=documents,
        taxonomies=taxonomies,
        riskgroups=risk_groups,
        risks=risk_objects,
    )
    return container


def get_risks() -> dict[str, str]:
    # Create an empty dataframe

    df = pd.DataFrame(
        columns=["cate-idx", "l2-name", "l3-name", "l4-name", "definition"]
    )
    row_list = []

    for row in ds:
        result = modify_paragraph(row["judge_prompt"])
        title = row["l4-name"]
        def_text = title + " " + result
        row_list.append(
            {
                "cate-idx": row["cate-idx"],
                "l2-name": row["l2-name"],
                "l3-name": row["l3-name"],
                "l4-name": row["l4-name"],
                "definition": def_text,
            }
        )

    return row_list


def modify_paragraph(paragraph):
    # Split the paragraph into sentences
    sentences = re.split(r"(?<=\.)\s+", paragraph)  # Splitting at sentence boundaries

    if len(sentences) < 2:
        return "Paragraph does not contain enough sentences."

    # Extract the second sentence
    second_sentence = sentences[1]

    # Modify the second sentence
    modified_sentence = re.sub(
        r"\byour task is to evaluate\b",
        "is defined as",
        second_sentence,
        flags=re.IGNORECASE,
    )
    modified_sentence = re.sub(
        r"\byour task is to determine\b",
        "is defined as",
        modified_sentence,
        flags=re.IGNORECASE,
    )

    return modified_sentence


def get_related_risks(related_risks_str: str, risks: dict[str, str]) -> list[str]:
    related_risks = related_risks_str.split(";")
    return [
        id
        for name, id in risks.items()
        for risk in related_risks
        if name == risk.strip()
    ]


export_file_path = os.path.join(DATA_DIR, "air_2024_data.yaml")

with open(export_file_path, "+tw", encoding="utf-8") as output_file:
    container = create_container_object()
    print(YAMLDumper().dumps(container), file=output_file)
    output_file.close()
