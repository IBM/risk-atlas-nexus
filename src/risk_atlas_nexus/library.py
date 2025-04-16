import os
import yaml
import json
from linkml_runtime import SchemaView
from linkml_runtime.dumpers import YAMLDumper
from typing import Optional, List, Dict

from importlib.metadata import version
from sssom_schema import Mapping

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import (
    Action,
    Risk,
    RiskControl,
    RiskIncident,
    RiskTaxonomy,
)
from risk_atlas_nexus.blocks.inference.templates import COT_TEMPLATE, AI_TASKS_TEMPLATE
from risk_atlas_nexus.blocks.risk_detector import AutoRiskDetector
from risk_atlas_nexus.blocks.risk_explorer import RiskExplorer
from risk_atlas_nexus.blocks.risk_mapping import RiskMapper
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.toolkit.data_utils import load_yamls_to_container
from risk_atlas_nexus.toolkit.error_utils import type_check, value_check
from risk_atlas_nexus.data import get_templates_path
from risk_atlas_nexus.toolkit.logging import configure_logger
from risk_atlas_nexus.blocks.inference.response_schema import LIST_OF_STR_SCHEMA
from risk_atlas_nexus.metadata_base import MappingMethod


logger = configure_logger(__name__)


class RiskAtlasNexus:
    """A RiskAtlasNexus object"""

    # Load the schema
    directory = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(directory, "ai_risk_ontology/schema/ai-risk-ontology.yaml")
    schema_view = SchemaView(yaml.safe_load(fn))

    def __init__(self, base_dir: str = None):
        """Create a new RiskAtlasNexus object

        Args:
            base_dir: str
                (Optional) add an alternative source of date
        """
        if base_dir is not None:
            if type(base_dir) != str:
                raise ValueError("Base directory must be a string", base_dir)
            if not os.path.isdir(base_dir):
                logger.error(f"Directory %s does not exist.", base_dir)
                raise FileNotFoundError("Base directory is not found", base_dir)

        ontology = load_yamls_to_container(base_dir)
        self._ontology = ontology
        self._risk_explorer = RiskExplorer(ontology)
        logger.info(f"Created RiskAtlasNexus instance. Base_dir: %s", base_dir)

    def export(cls, export_path):
        """Export RiskAtlasNexus configuration to file.

        Args:
            export_path: str
                The path to the directory where the artifact will be exported to.

        """
        if not os.path.isdir(export_path):
            logger.error(f"Directory %s does not exist.", export_path)
            raise FileNotFoundError("Export directory is not found", export_path)

        export_file_path = os.path.join(export_path, "ai-risk-ontology.yaml")

        with open(export_file_path, "+tw", encoding="utf-8") as output_file:
            print(YAMLDumper().dumps(cls._ontology), file=output_file)
            output_file.close()

    @classmethod
    def get_schema(cls):
        """Get schema

        Returns:
            schema
        """
        return cls.schema_view

    @classmethod
    def get_version(cls):
        """Get library version

        Returns:
            dict: Version number
        """
        response = {"version": version("risk_atlas_nexus")}
        return response

    def get_all_risks(cls, taxonomy=None):
        """Get all risk definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Risk]
                Result containing a list of AI risks
        """
        type_check("<RANEACF44A7E>", str, allow_none=True, taxonomy=taxonomy)

        risk_instances = cls._risk_explorer.get_all_risks(taxonomy)
        return risk_instances

    def get_risk(cls, tag=None, id=None, name=None, taxonomy=None):
        """Get risk definition from the LinkML, filtered by risk atlas id, tag, name

        Args:
            id: (Optional) str
                The string ID identifying the risk
            tag: (Optional) str
                The string tag identifying the risk
            name: (Optional) str
                The string name identifying the risk
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Risk
                Result containing a list of AI risks
        """
        type_check(
            "<RAND62C1B3AE>",
            str,
            allow_none=True,
            tag=tag,
            id=id,
            name=name,
            taxonomy=taxonomy,
        )
        value_check(
            "<RAN14D4D967E>",
            tag or id or name,
            "Please provide tag, id, or name",
        )

        risk: Risk | None = cls._risk_explorer.get_risk(
            tag=tag, id=id, name=name, taxonomy=taxonomy
        )
        return risk

    def get_related_risks(cls, risk=None, tag=None, id=None, name=None, taxonomy=None):
        """Get related risks from the LinkML, filtered by risk id, tag, or name

        Args:
            risk: (Optional) Risk
                The risk
            id: (Optional) str
                The string ID identifying the risk
            tag: (Optional) str
                The string tag identifying the risk
            name: (Optional) str
                The string name identifying the risk
            taxonomy: str
                (Optional) The string label for a taxonomy
        Returns:
            List[str]
                Result containing a list of AI risk IDs
        """
        type_check("<RAN283B72CAE>", Risk, allow_none=True, risk=risk)
        type_check(
            "<RANC9FDCC45E>",
            str,
            allow_none=True,
            tag=tag,
            id=id,
            name=name,
            taxonomy=taxonomy,
        )
        value_check(
            "<RAN0748ECB7E>",
            risk or tag or id or name,
            "Please provide tag, id, or name",
        )

        related_risk_instances = cls._risk_explorer.get_related_risks(
            risk=risk, tag=tag, id=id, name=name, taxonomy=taxonomy
        )
        return related_risk_instances

    def get_related_actions(
        cls, risk=None, tag=None, id=None, name=None, taxonomy=None
    ):
        """Get actions for a risk definition from the LinkML.  The risk is identified by risk id, tag, or name

        Args:
            risk: (Optional) Risk
                The risk
            id: (Optional) str
                The string ID identifying the risk
            tag: (Optional) str
                The string tag identifying the risk
            name: (Optional) str
                The string name identifying the risk
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Risk
                Result containing a list of AI actions
        """
        type_check("<RANEDB39EABE>", Risk, allow_none=True, risk=risk)
        type_check(
            "<RANC49E332BE>",
            str,
            allow_none=True,
            tag=tag,
            id=id,
            name=name,
            taxonomy=taxonomy,
        )
        value_check(
            "<RAN7154EE0FE>",
            risk or tag or id or name,
            "Please provide risk, tag, id, or name",
        )

        actions = cls._risk_explorer.get_related_actions(
            risk=risk, tag=tag, id=id, name=name, taxonomy=taxonomy
        )
        return actions

    def get_all_actions(cls, taxonomy=None):
        """Get all action definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Action]
                Result containing a list of AI actions
        """
        type_check("<RAN1C9A35ADE>", str, allow_none=True, taxonomy=taxonomy)

        action_instances: list[Action] = cls._risk_explorer.get_all_actions(taxonomy)
        return action_instances

    def get_action_by_id(cls, id, taxonomy=None):
        """Get an action definition from the LinkML, filtered by action id

        Args:
            id: str
                The string id identifying the action
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Action
                Result containing an action
        """
        type_check("<RAN66203B1FE>", str, allow_none=False, id=id)
        type_check("<RAN869039B6E>", str, allow_none=True, taxonomy=taxonomy)

        action: Action | None = cls._risk_explorer.get_action_by_id(id=id)
        return action

    def get_related_risk_controls(
        cls, risk=None, tag=None, id=None, name=None, taxonomy=None
    ):
        """Get related risk controls for a risk definition from the LinkML.  The risk is identified by risk id, tag, or name

        Args:
            risk: (Optional) Risk
                The risk
            id: (Optional) str
                The string ID identifying the risk
            tag: (Optional) str
                The string tag identifying the risk
            name: (Optional) str
                The string name identifying the risk
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Risk
                Result containing a list of AI actions
        """
        type_check("<RAN4E03158FE>", Risk, allow_none=True, risk=risk)
        type_check(
            "<RAN55784808E>",
            str,
            allow_none=True,
            tag=tag,
            id=id,
            name=name,
            taxonomy=taxonomy,
        )
        value_check(
            "<RAN5DCADF94E>",
            risk or tag or id or name,
            "Please provide risk, tag, id, or name",
        )

        risk_controls = cls._risk_explorer.get_related_risk_controls(
            risk=risk, tag=tag, id=id, name=name, taxonomy=taxonomy
        )
        return risk_controls

    def get_all_risk_controls(cls, taxonomy=None):
        """Get all risk control definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[RiskControl]
                Result containing a list of RiskControls
        """
        type_check("<RAN129A1692E>", str, allow_none=True, taxonomy=taxonomy)

        risk_control_instances: list[RiskControl] = (
            cls._risk_explorer.get_all_risk_controls(taxonomy)
        )
        return risk_control_instances

    def get_risk_control(cls, id=None, taxonomy=None):
        """Get an action definition from the LinkML, filtered by risk control id

        Args:
            id: str
                The string id identifying the risk control
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Action
                Result containing a risk control.
        """
        type_check("<RAN9785FFE3E>", str, allow_none=False, id=id)
        type_check("<RAN5A157049E>", str, allow_none=True, taxonomy=taxonomy)

        risk_control: RiskControl | None = cls._risk_explorer.get_risk_control(id=id)
        return risk_control

    def identify_risks_from_usecases(
        cls,
        usecases: List[str],
        inference_engine: InferenceEngine,
        taxonomy: Optional[str] = None,
    ) -> List[List[Risk]]:
        """Identify potential risks from a usecase description

        Args:
            usecases (List[str]):
                A List of strings describing AI usecases
            inference_engine (InferenceEngine):
                An LLM inference engine to infer risks from the usecases.
            taxonomy (str):
                The string label for a taxonomy

        Returns:
            List[List[Risk]]:
                Result containing a list of risks
        """
        type_check(
            "<RANE023314BE>",
            InferenceEngine,
            allow_none=False,
            inference_engine=inference_engine,
        )
        type_check("<RANB72CAE6EE>", str, allow_none=True, taxonomy=taxonomy)
        value_check(
            "<RAN4717CF18E>",
            usecases or inference_engine,
            "Please provide usecases and inference_engine",
        )

        risk_detector = AutoRiskDetector.create(
            cls._ontology, inference_engine=inference_engine, taxonomy=taxonomy
        )

        return risk_detector.detect(usecases)

    def get_all_taxonomies(cls):
        """Get all taxonomy definitions from the LinkML

        Returns:
            List[RiskTaxonomy]
                Result containing a list of AI Risk taxonomies
        """
        taxonomy_instances: list[RiskTaxonomy] = cls._risk_explorer.get_all_taxonomies()
        return taxonomy_instances

    def get_taxonomy_by_id(cls, id):
        """Get taxonomy definitions from the LinkML filtered by taxonomy id

        Args:
            id: str
                The string id for a taxonomy

        Returns:
            RiskTaxonomy
                An AI Risk taxonomy
        """
        type_check("<RANBFB574E3E>", str, allow_none=False, id=id)

        taxonomy: RiskTaxonomy | None = cls._risk_explorer.get_taxonomy_by_id(id)
        return taxonomy

    def generate_zero_shot_output(
        cls,
        inference_engine: InferenceEngine,
        usecase: str,
        questions: List[str],
    ):
        """Get prediction using the zero shot approach.

        Args:
            usecase (str): A string describing an AI usecase
            inference_engine (InferenceEngine):
                An LLM inference engine to predict the output based on the given use case.
            questions (List[str]): A list of questions.
                Check example below.
                ```
                [
                    "In which environment is the system used?",
                ]
                ```

        Returns:
            List[str]: List of LLM predictions.
        """
        type_check(
            "<RANF7EFFADAE>",
            InferenceEngine,
            allow_none=False,
            inference_engine=inference_engine,
        )
        type_check("<RANB9FDEA04E>", str, allow_none=False, usecase=usecase)
        type_check("<RANF7256EC3E>", List, allow_none=False, questions=questions)
        value_check(
            "<RANC49F00D3E>",
            inference_engine and questions,
            "Please provide questions and inference_engine",
        )

        prompts = [
            inference_engine.prepare_prompt(
                prompt_template=COT_TEMPLATE,
                usecase=usecase,
                question=question,
                examples=None,
            )
            for question in questions
        ]
        return [
            result.prediction
            for result in inference_engine.generate(
                prompts, postprocessors=["clean_output"]
            )
        ]

    def generate_few_shot_output(
        cls,
        inference_engine: InferenceEngine,
        usecase: str,
        cot_data: List[Dict],
    ):
        """Get prediction using the few shot (Chain of Thought) examples.

        Args:
            usecase (str): A string describing an AI usecase
            inference_engine (InferenceEngine):
                An LLM inference engine to predict the output based on the given use case.
            cot_data (List[Dict]): Chain of Thought data.
                Each question is associated with a list of example intents and
                corresponding answers. Check example JSON below.
                ```
                [
                    {
                        "question": "In which environment is the system used?",
                        "examples": {
                            "intents": [
                                "Find patterns in healthcare insurance claims",
                            ]
                            "answers": [
                                "Insurance companies, government agencies, or other organizations responsible for reimbursing healthcare providers. Explanation: Healthcare payers need to efficiently process and reimburse claims while minimizing errors and disputes. By identifying patterns in claims data, they can automate routine tasks, detect potential errors or anomalies, and improve overall payment accuracy.",
                            ],
                        }
                    }
                ]
                ```

        Returns:
            List[str]: List of LLM predictions.
        """

        prompts = []
        for data in cot_data:
            assert (
                "examples" in data and len(data["examples"]) > 0
            ), f"When using the Few shot API, `cot_data` must include examples. Question: [{data['question']}] does not have examples."

            assert len(data["examples"]["answers"]) == len(
                data["examples"]["intents"]
            ), f"Few shot intents and answers should be the same length for Question: [{data['question']}]"

            prompts.append(
                inference_engine.prepare_prompt(
                    prompt_template=COT_TEMPLATE,
                    usecase=usecase,
                    question=data["question"],
                    examples=(
                        [
                            {"answer": answer, "usecase": intent}
                            for answer, intent in zip(
                                data["examples"]["answers"], data["examples"]["intents"]
                            )
                        ]
                    ),
                )
            )

        return [
            result.prediction
            for result in inference_engine.generate(
                prompts, postprocessors=["clean_output"]
            )
        ]

    def identify_ai_tasks_from_usecases(
        cls,
        usecases: List[str],
        inference_engine: InferenceEngine,
    ) -> List[List[str]]:
        """Identify potential risks from a usecase description

        Args:
            usecases (List[str]):
                A List of strings describing AI usecases
            inference_engine (InferenceEngine):
                An LLM inference engine to identify AI tasks from usecases.

        Returns:
            List[List[str]]:
                Result containing a list of AI tasks
        """
        type_check(
            "<RAN3B9CD886E>",
            InferenceEngine,
            allow_none=False,
            inference_engine=inference_engine,
        )
        type_check("<RAN4CDA6852E>", List, allow_none=False, usecases=usecases)
        value_check(
            "<RAN0E435F50E>",
            inference_engine and usecases,
            "Please provide usecases and inference_engine",
        )

        with open(os.path.join(get_templates_path(), "hf_ai_tasks.json")) as f:
            hf_ai_tasks = json.load(f)
        prompts = [
            inference_engine.prepare_prompt(
                prompt_template=AI_TASKS_TEMPLATE,
                usecase=usecase,
                hf_ai_tasks=hf_ai_tasks,
                limit=len(hf_ai_tasks),
            )
            for usecase in usecases
        ]

        LIST_OF_STR_SCHEMA["items"]["enum"] = [
            task["task_label"] for task in hf_ai_tasks
        ]
        return [
            result.prediction
            for result in inference_engine.generate(
                prompts,
                response_format=LIST_OF_STR_SCHEMA,
                postprocessors=["list_of_str"],
            )
        ]

    def generate_proposed_mappings(
        cls,
        new_risks: List[Risk],
        existing_risks: List[Risk],
        inference_engine: InferenceEngine,
        new_prefix: str,  
        mapping_method: MappingMethod = MappingMethod.SEMANTIC
    ) -> List[Mapping]:
        """Identify mappings between a new set of risks and risks that exist in the Risk Atlas

        Args:
            new_risks: List[Risk]
                A new set of risks
            existing_risks: List[Risk]
                Secondary list, this should be the list of existing risks in RAN
            inference_engine: (Optional)Union[InferenceEngine | None]:
                An LLM inference engine to infer risks from the use cases.
            new_prefix: str
                The CURIE prefix for the new list of risks
            mapping_method: MappingMethod
                The possible values for type of risk mapping method

        Returns:
            List[Mapping]
                Result containing a list of mappings
        """
        type_check(
            "<RAN28959363E>",
            InferenceEngine,
            allow_none=True,
            inference_engine=inference_engine,
        )
        value_check(
            "<RAN85167315E>",
            new_risks and existing_risks,
            "Please provide new_risks and existing_risks",
        )
        risk_mapper = RiskMapper(new_risks=new_risks, existing_risks=existing_risks, inference_engine=inference_engine, 
                                 new_prefix=new_prefix, mapping_method=mapping_method
        )

        return risk_mapper.generate(new_risks=new_risks, existing_risks=existing_risks, inference_engine=inference_engine, new_prefix=new_prefix, mapping_method=mapping_method)


    def get_risk_incidents(cls, taxonomy: Optional[str] = None):
        """Get risk incident instances, optionally filtered by taxonomy

        Returns:
            List[RiskIncident]
                Result containing a list of AI Risk Incidents
        """
        type_check("<RAN04811131E>", str, allow_none=True, taxonomy=taxonomy)

        risk_incident_instances: List[RiskIncident] = cls._risk_explorer.get_risk_incidents(taxonomy=taxonomy)
        return risk_incident_instances
    
    def get_risk_incident(cls, id=None, taxonomy=None):
        """Get an risk incident instance filtered by risk incident id

        Args:
            id: str
                The string id identifying the risk incident
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            RiskIncident
                Result containing a risk incident.
        """
        type_check("<RAN97353068E>", str, allow_none=False, id=id)
        type_check("<RAN38198685E>", str, allow_none=True, taxonomy=taxonomy)

        risk_incident: RiskIncident | None = cls._risk_explorer.get_risk_incident(id=id)
        return risk_incident
    
    def get_related_risk_incidents(cls, risk=None, risk_id=None, taxonomy=None):
        """Get related risk incident filtered by risk id

        Args:
            risk: (Optional) Risk
                The risk
            risk_id: (Optional) str
                The string ID identifying the risk
            taxonomy: str
                (Optional) The string label for a taxonomy
        Returns:
            List[RiskIncident]
                Result containing a list of AI risk incidents
        """
        type_check("<RAN40791379E>", Risk, allow_none=True, risk=risk)
        type_check(
            "<RANC9FDCC45E>",
            str,
            allow_none=True,
            risk_id=risk_id,
            taxonomy=taxonomy,
        )
        value_check(
            "<RAN79007538E>",
            risk or id,
            "Please provide risk or id",
        )

        related_risk_incidents = cls._risk_explorer.get_related_risk_incidents(
            risk=risk, risk_id=risk_id, taxonomy=taxonomy
        )
        return related_risk_incidents