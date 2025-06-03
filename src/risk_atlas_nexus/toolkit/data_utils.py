import glob
import os

from linkml_runtime.loaders import yaml_loader

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Container
from risk_atlas_nexus.data import get_data_path
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)


def load_yamls_to_container(base_dir):
    """Function to load the RiskAtlasNexus with data

    Args:
        base_dir: str
            (Optional) user defined base directory path

    Returns:
        YAMLRoot instance of the Container class
    """

    # Get system yaml data path
    system_data_path = get_data_path()

    master_yaml_files = []
    for yaml_dir in [system_data_path, base_dir]:
        # Include YAML files from the user defined `base_dir` if exist.
        if yaml_dir is not None:
            master_yaml_files.extend(
                glob.glob(os.path.join(yaml_dir, "**", "*.yaml"), recursive=True)
            )

    yml_items_result = {}
    for yaml_file in master_yaml_files:
        try:
            yml_items = yaml_loader.load_as_dict(source=yaml_file)
            for ontology_class, instances in yml_items.items():
                yml_items_result.setdefault(ontology_class, []).extend(instances)
        except Exception as e:
            logger.info(f"YAML ignored: {yaml_file}. Failed to load. {e}")

    # TODO: generalise this to cover all ontology classes

    # combine any risk entries which share the same id, for example a risk, and a secondary entry for a mapping
    combine_risks = {}

    for risk in yml_items_result["risks"]:
        risk_id = risk["id"]

        if risk_id not in combine_risks:
            combine_risks[risk_id] = {"id": risk_id}

        for key, value in risk.items():
            if key != "id":
                if key not in combine_risks[risk_id]:
                    combine_risks[risk_id][key] = value
                else:
                    if combine_risks[risk_id][key] is not None:
                        combine_risks[risk_id][key] = [
                            *combine_risks[risk_id][key],
                            *value,
                        ]
                    else:
                        combine_risks[risk_id][key] = value

    yml_items_result["risks"] = list(combine_risks.values())

    ontology = yaml_loader.load_any(
        source=yml_items_result,
        target_class=Container,
    )

    return ontology
