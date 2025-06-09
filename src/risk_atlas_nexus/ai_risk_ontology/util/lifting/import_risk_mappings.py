# Standard Library
from os import listdir
from os.path import isfile, join
from pathlib import Path

# Third Party
from linkml_runtime.dumpers import YAMLDumper
from pydantic import BaseModel
from sssom.parsers import parse_sssom_table

# Local
from risk_atlas_nexus.ai_risk_ontology import Container, Risk
from risk_atlas_nexus.toolkit.logging import configure_logger


MAP_DIR = "src/risk_atlas_nexus/data/mappings/"
DATA_DIR = "src/risk_atlas_nexus/data/knowledge_graph/mappings/"

logger = configure_logger(__name__)


class RiskMap(BaseModel):
    src_risk_id: str
    target_risk_id: str
    relationship: str

    def __init__(self, src_risk_id: str, target_risk_id: str, relationship: str):
        src_id = src_risk_id.split(":")[-1]
        target_id = target_risk_id.split(":")[-1]

        super().__init__(
            src_risk_id=src_id,
            target_risk_id=target_id,
            relationship=relationship,
        )


def process_mapping_from_tsv_to_risk_mapping(file_name):
    tsv_file_name = join(MAP_DIR, file_name)
    mapping_set_df = parse_sssom_table(file_path=tsv_file_name)
    ms = mapping_set_df.to_mapping_set()
    risk_maps = [
        RiskMap(
            **{
                "src_risk_id": item["subject_id"],
                "target_risk_id": item["object_id"],
                "relationship": item["predicate_id"],
            }
        )
        for item in ms.mappings
        if item["predicate_id"] != "noMatch"
    ]
    return risk_maps


def process_mappings_to_risks(risk_maps):
    output_risks = []
    for rm in risk_maps:

        s_id = rm.src_risk_id
        o_id = rm.target_risk_id
        risk = Risk(id=s_id)
        risk_for_inverse = Risk(id=o_id)

        relationship = rm.relationship
        if relationship == "skos:closeMatch":
            risk.closeMatch = [o_id]
            risk_for_inverse.closeMatch = [s_id]
        elif relationship == "skos:exactMatch":
            risk.exactMatch = [o_id]
            risk_for_inverse.exactMatch = [s_id]
        elif relationship == "skos:broadMatch":
            risk.broadMatch = [o_id]
            risk_for_inverse.narrowMatch = [s_id]
        elif relationship == "skos:narrowMatch":
            risk.narrowMatch = [o_id]
            risk_for_inverse.broadMatch = [s_id]
        elif relationship == "skos:relatedMatch":
            risk.relatedMatch = [o_id]
            risk_for_inverse.relatedMatch = [s_id]
        else:
            logger.info("Unparseable predicate_id: %s", relationship)

        output_risks.append(risk)
        output_risks.append(risk_for_inverse)

    return output_risks


def write_to_file(output_risks, output_file):
    with open(output_file, "+tw", encoding="utf-8") as output_file:
        print(YAMLDumper().dumps(Container(risks=output_risks)), file=output_file)
        output_file.close()


if __name__ == "__main__":
    logger.info(f"Processing mapping files in : %s", MAP_DIR)
    mapping_files = [
        file_name for file_name in listdir(MAP_DIR) if (isfile(join(MAP_DIR, file_name) and not file_name.endswith(".md")))
    ]
    for file_name in mapping_files:
        output_file = DATA_DIR + Path(file_name).stem + "_from_tsv_data.yaml"
        rs = process_mapping_from_tsv_to_risk_mapping(file_name)
        logger.info(f"Processed file: %s, %s valid entries", file_name, len(rs))
        output_risks = process_mappings_to_risks(rs)
        write_to_file(output_risks, output_file)
