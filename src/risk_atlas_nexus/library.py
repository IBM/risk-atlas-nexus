import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template
from linkml_runtime import SchemaView
from linkml_runtime.dumpers import YAMLDumper
from sssom_schema import Mapping


# workaround for txtai
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import (
    Action,
    AiEval,
    BenchmarkMetadataCard,
    Dataset,
    Documentation,
    Risk,
    RiskControl,
    RiskIncident,
    RiskTaxonomy,
)
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.blocks.prompt_builder import (
    FewShotPromptBuilder,
    ZeroShotPromptBuilder,
)
from risk_atlas_nexus.blocks.prompt_response_schema import (
    DOMAIN_TYPE_SCHEMA,
    LIST_OF_STR_SCHEMA,
    QUESTIONNAIRE_OUTPUT_SCHEMA,
)
from risk_atlas_nexus.blocks.prompt_templates import (
    AI_TASKS_TEMPLATE,
    QUESTIONNAIRE_COT_TEMPLATE,
)
from risk_atlas_nexus.blocks.risk_categorization.severity import RiskSeverityCategorizer
from risk_atlas_nexus.blocks.risk_detector import GenericRiskDetector
from risk_atlas_nexus.blocks.risk_explorer import RiskExplorer
from risk_atlas_nexus.blocks.risk_mapping import RiskMapper
from risk_atlas_nexus.data import load_resource
from risk_atlas_nexus.metadata_base import MappingMethod
from risk_atlas_nexus.toolkit.data_utils import load_yamls_to_container
from risk_atlas_nexus.toolkit.error_utils import type_check, value_check
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)


class RiskAtlasNexus:
    """A RiskAtlasNexus object"""

    # Load the schema
    directory = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(
        directory,
        "ai_risk_ontology/schema/ai-risk-ontology.yaml",
    )
    schema_view = SchemaView(yaml.safe_load(fn))

    def __init__(self, base_dir: str = None):
        """Create a new RiskAtlasNexus object

        Args:
            base_dir: str
                (Optional) add an alternative source of date
        """
        if base_dir is not None:
            if type(base_dir) != str:
                raise ValueError(
                    "Base directory must be a string",
                    base_dir,
                )
            if not os.path.isdir(base_dir):
                logger.error(
                    f"Directory %s does not exist.",
                    base_dir,
                )
                raise FileNotFoundError(
                    "Base directory is not found",
                    base_dir,
                )

        ontology = load_yamls_to_container(base_dir)
        self._ontology = ontology
        self._risk_explorer = RiskExplorer(ontology)
        logger.info(
            f"Created RiskAtlasNexus instance. Base_dir: %s",
            base_dir,
        )

    def export(cls, export_path):
        """Export RiskAtlasNexus configuration to file.

        Args:
            export_path: str
                The path to the directory where the artifact will be exported to.

        """
        if not os.path.isdir(export_path):
            logger.error(
                f"Directory %s does not exist.",
                export_path,
            )
            raise FileNotFoundError(
                "Export directory is not found",
                export_path,
            )

        export_file_path = os.path.join(export_path, "ai-risk-ontology.yaml")

        with open(
            export_file_path,
            "+tw",
            encoding="utf-8",
        ) as output_file:
            print(
                YAMLDumper().dumps(cls._ontology),
                file=output_file,
            )
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
        type_check(
            "<RANEACF44A7E>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )

        risk_instances = cls._risk_explorer.get_all_risks(taxonomy)
        return risk_instances

    def get_risk(
        cls,
        tag=None,
        id=None,
        name=None,
        taxonomy=None,
    ):
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
            tag=tag,
            id=id,
            name=name,
            taxonomy=taxonomy,
        )
        return risk

    def get_related_risks(
        cls,
        risk=None,
        tag=None,
        id=None,
        name=None,
        taxonomy=None,
    ):
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
        type_check(
            "<RAN283B72CAE>",
            Risk,
            allow_none=True,
            risk=risk,
        )
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
            risk=risk,
            tag=tag,
            id=id,
            name=name,
            taxonomy=taxonomy,
        )
        return related_risk_instances

    def get_related_actions(
        cls,
        risk=None,
        tag=None,
        id=None,
        name=None,
        taxonomy=None,
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
        type_check(
            "<RANEDB39EABE>",
            Risk,
            allow_none=True,
            risk=risk,
        )
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
            risk=risk,
            tag=tag,
            id=id,
            name=name,
            taxonomy=taxonomy,
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
        type_check(
            "<RAN1C9A35ADE>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )

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
        type_check(
            "<RAN66203B1FE>",
            str,
            allow_none=False,
            id=id,
        )
        type_check(
            "<RAN869039B6E>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )

        action: Action | None = cls._risk_explorer.get_action_by_id(id=id)
        return action

    def get_related_risk_controls(
        cls,
        risk=None,
        tag=None,
        id=None,
        name=None,
        taxonomy=None,
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
        type_check(
            "<RAN4E03158FE>",
            Risk,
            allow_none=True,
            risk=risk,
        )
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
            risk=risk,
            tag=tag,
            id=id,
            name=name,
            taxonomy=taxonomy,
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
        type_check(
            "<RAN129A1692E>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )

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
        type_check(
            "<RAN9785FFE3E>",
            str,
            allow_none=False,
            id=id,
        )
        type_check(
            "<RAN5A157049E>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )

        risk_control: RiskControl | None = cls._risk_explorer.get_risk_control(id=id)
        return risk_control

    def identify_risks_from_usecases(
        cls,
        usecases: List[str],
        inference_engine: InferenceEngine,
        taxonomy: Optional[str] = None,
        cot_examples: Optional[Dict[str, List]] = None,
        max_risk: Optional[int] = None,
    ) -> List[List[Risk]]:
        """Identify potential risks from a usecase description

        Args:
            usecases (List[str]):
                A List of strings describing AI usecases
            inference_engine (InferenceEngine):
                An LLM inference engine to infer risks from the usecases.
            taxonomy (str, optional):
                The string label for a taxonomy. Default to None.
            cot_examples (Dict[str, List], optional):
                The Chain of Thought (CoT) examples to use in the risk identification.
                The example template is available at src/risk_atlas_nexus/data/templates/risk_generation_cot.json.
                Assign the ID of the taxonomy you wish to use as the key for CoT examples. Providing this value
                will override the CoT examples present in the template master. Default to None.
            max_risk (int, optional):
                The maximum number of risks to extract. Pass None to allow the inference engine to determine the number of risks. Defaults to None.

        Returns:
            List[List[Risk]]:
                Result containing a list of risks
        """
        type_check(
            "<RANE023314BE>",
            List,
            allow_none=False,
            usecases=usecases,
        )
        type_check(
            "<RANE023314BE>",
            InferenceEngine,
            allow_none=False,
            inference_engine=inference_engine,
        )
        type_check(
            "<RANB72CAE6EE>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )
        type_check(
            "<RAN80975498E>",
            int,
            allow_none=True,
            max_risk=max_risk,
        )
        value_check(
            "<RAN4717CF18E>",
            all([isinstance(usecase, str) for usecase in usecases]),
            "Usecases must be a list of string.",
        )

        risk_detector = GenericRiskDetector(
            cls._ontology,
            inference_engine=inference_engine,
            taxonomy=taxonomy,
            cot_examples=cot_examples,
            max_risk=max_risk,
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
        type_check(
            "<RANBFB574E3E>",
            str,
            allow_none=False,
            id=id,
        )

        taxonomy: RiskTaxonomy | None = cls._risk_explorer.get_taxonomy_by_id(id)
        return taxonomy

    def generate_zero_shot_risk_questionnaire_output(
        cls,
        usecase: str,
        risk_questionnaire: List[Dict[str, str]],
        inference_engine: InferenceEngine,
        verbose=True,
    ):
        """Get prediction using the zero shot approach.

        Args:
            usecase (str): A string describing an AI usecase
            risk_questionnaire: List[Dict[str, str]]: A risk questionnaire
                Check example below.
                ```
                [
                    "In which environment is the system used?",
                ]
                ```
            inference_engine (InferenceEngine):
                An LLM inference engine to predict the output based on the given use case.

        Returns:
            List[str]: List of LLM predictions.
        """
        type_check(
            "<RANF7EFFADAE>",
            InferenceEngine,
            allow_none=False,
            inference_engine=inference_engine,
        )
        type_check(
            "<RANB9FDEA04E>",
            str,
            allow_none=False,
            usecase=usecase,
        )
        type_check(
            "<RANF7256EC3E>",
            List,
            allow_none=False,
            questions=risk_questionnaire,
        )
        value_check(
            "<RANC49F00D3E>",
            inference_engine and risk_questionnaire,
            "Please provide questions and inference_engine",
        )

        # Extract only questions
        risk_questionnaire = [
            question_data["question"] for question_data in risk_questionnaire
        ]

        # Prepare zero shots inference prompts
        prompts = [
            ZeroShotPromptBuilder(
                QUESTIONNAIRE_COT_TEMPLATE,
            ).build(usecase=usecase, question=question)
            for question in risk_questionnaire
        ]

        # Invoke inference service
        return inference_engine.generate(
            prompts,
            response_format=QUESTIONNAIRE_OUTPUT_SCHEMA,
            postprocessors=["json_object"],
            verbose=verbose,
        )

    def generate_few_shot_risk_questionnaire_output(
        cls,
        usecase: str,
        risk_questionnaire: List[Dict[str, Any]],
        inference_engine: InferenceEngine,
        verbose=True,
    ):
        """Get prediction using the few shot (Chain of Thought) examples.

        Args:
            usecase (str): A string describing an AI usecase
            risk_questionnaire (List[Dict]): Chain of Thought data for risk questionnaire.
                Each question is associated with a list of example intents and
                corresponding answers. Check example JSON below.
                ```
                [
                    {
                        "question": "In which environment is the system used?",
                        "examples": [
                            "intent": "Find patterns in healthcare insurance claims",
                            "answer": "Insurance Claims Processing or Risk Management or Data Analytics",
                            "explanation": "The system might be used by an insurance company's claims processing department to analyze and identify patterns in healthcare insurance claims."
                        ]
                    }
                ]
            inference_engine (InferenceEngine):
                An LLM inference engine to predict the output based on the given use case.
            filter_cot_data_by (Dict[str, str]):
                A dictionary to filter CoT examples with key as CoT field and value as filter string.
                ```

        Returns:
            List[str]: List of LLM predictions.
        """
        type_check(
            "<RAN19989483E>",
            InferenceEngine,
            allow_none=False,
            inference_engine=inference_engine,
        )
        type_check(
            "<RAN17812927E>",
            str,
            allow_none=False,
            usecase=usecase,
        )
        type_check(
            "<RAN46376875E>",
            List,
            allow_none=False,
            questions=risk_questionnaire,
        )
        value_check(
            "<RAN59638961E>",
            inference_engine and risk_questionnaire,
            "Please provide risk_questionnaire_cot and inference_engine",
        )

        assert (
            risk_questionnaire and len(risk_questionnaire) > 0
        ), "`Chain of Thought (risk_questionnaire_cot)` data cannot be None or empty."

        # Prepare few shots inference prompts from CoT Data
        prompts = [
            FewShotPromptBuilder(QUESTIONNAIRE_COT_TEMPLATE).build(
                cot_examples=question_data["cot_examples"],
                usecase=usecase,
                question=question_data["question"],
            )
            for question_data in risk_questionnaire
        ]

        # Invoke inference service
        return inference_engine.generate(
            prompts,
            response_format=QUESTIONNAIRE_OUTPUT_SCHEMA,
            postprocessors=["json_object"],
            verbose=verbose,
        )

    def identify_ai_tasks_from_usecases(
        cls, usecases: List[str], inference_engine: InferenceEngine, verbose=True
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

        # Load HF tasks from the template dir
        hf_ai_tasks = load_resource("hf_ai_tasks.json")

        # Populate schema items
        json_schema = dict(LIST_OF_STR_SCHEMA)
        json_schema["items"]["enum"] = [task["task_label"] for task in hf_ai_tasks]

        # Invoke inference service
        return inference_engine.generate(
            prompts=[
                Template(AI_TASKS_TEMPLATE).render(
                    usecase=usecase, hf_ai_tasks=hf_ai_tasks, limit=len(hf_ai_tasks)
                )
                for usecase in usecases
            ],
            response_format=json_schema,
            postprocessors=["list_of_str"],
            verbose=verbose,
        )

    def generate_proposed_mappings(
        cls,
        new_risks: List[Risk],
        existing_risks: List[Risk],
        inference_engine: InferenceEngine,
        new_prefix: str,
        mapping_method: MappingMethod = MappingMethod.SEMANTIC,
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
        risk_mapper = RiskMapper(
            new_risks=new_risks,
            existing_risks=existing_risks,
            inference_engine=inference_engine,
            new_prefix=new_prefix,
            mapping_method=mapping_method,
        )

        return risk_mapper.generate(
            new_risks=new_risks,
            existing_risks=existing_risks,
            inference_engine=inference_engine,
            new_prefix=new_prefix,
            mapping_method=mapping_method,
        )

    def get_risk_incidents(cls, taxonomy: Optional[str] = None):
        """Get risk incident instances, optionally filtered by taxonomy

        Returns:
            List[RiskIncident]
                Result containing a list of AI Risk Incidents
        """
        type_check(
            "<RAN04811131E>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )

        risk_incident_instances: List[RiskIncident] = (
            cls._risk_explorer.get_risk_incidents(taxonomy=taxonomy)
        )
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
        type_check(
            "<RAN97353068E>",
            str,
            allow_none=False,
            id=id,
        )
        type_check(
            "<RAN38198685E>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )

        risk_incident: RiskIncident | None = cls._risk_explorer.get_risk_incident(id=id)
        return risk_incident

    def get_related_risk_incidents(
        cls,
        risk=None,
        risk_id=None,
        taxonomy=None,
    ):
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
        type_check(
            "<RAN40791379E>",
            Risk,
            allow_none=True,
            risk=risk,
        )
        type_check(
            "<RANC9FDCC45E>",
            str,
            allow_none=True,
            risk_id=risk_id,
            taxonomy=taxonomy,
        )
        value_check(
            "<RAN79007538E>",
            risk or risk_id,
            risk or risk_id,
            "Please provide risk or id",
        )

        related_risk_incidents = cls._risk_explorer.get_related_risk_incidents(
            risk=risk,
            risk_id=risk_id,
            taxonomy=taxonomy,
        )
        return related_risk_incidents

    def get_all_evaluations(cls, taxonomy=None):
        """Get all evaluation definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[RiskControl]
                Result containing a list of AiEval
        """
        type_check("<RAN18094995E>", str, allow_none=True, taxonomy=taxonomy)

        evaluation_instances: list[AiEval] = cls._risk_explorer.get_all_evaluations(
            taxonomy
        )
        return evaluation_instances

    def get_evaluation(cls, id=None, taxonomy=None):
        """Get an evaluation definition from the LinkML, filtered by id

        Args:
            id: str
                The string id identifying the evaluation
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Action
                Result containing an evaluation.
        """
        type_check("<RAN84465757E>", str, allow_none=False, id=id)
        type_check("<RAN29906222E>", str, allow_none=True, taxonomy=taxonomy)

        evaluation: AiEval | None = cls._risk_explorer.get_evaluation(id=id)
        return evaluation

    def get_related_evaluations(cls, risk=None, risk_id=None, taxonomy=None):
        """Get related evaluations filtered by risk id

        Args:
            risk: (Optional) Risk
                The risk
            risk_id: (Optional) str
                The string ID identifying the risk
            taxonomy: str
                (Optional) The string label for a taxonomy
        Returns:
            List[AiEval]
                Result containing a list of AI evaluations
        """
        type_check("<RAN04616807E>", Risk, allow_none=True, risk=risk)
        type_check(
            "<RAN05640166E>",
            str,
            allow_none=True,
            risk_id=risk_id,
            taxonomy=taxonomy,
        )
        value_check(
            "<RAN39630388E>",
            risk or risk_id,
            "Please provide risk or id",
        )

        related_evaluations = cls._risk_explorer.get_related_evaluations(
            risk=risk, risk_id=risk_id, taxonomy=taxonomy
        )
        return related_evaluations

    def get_benchmark_metadata_cards(cls, risk=None, risk_id=None, taxonomy=None):
        """Get all benchmark metadata definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[BenchmarkMetadataCard]
                Result containing a list of BenchmarkMetadataCards
        """
        type_check(
            "<RAN07894687E>",
            str,
            allow_none=True,
            risk_id=risk_id,
            taxonomy=taxonomy,
        )
        type_check("<RAN30190075E>", Risk, allow_none=True, risk=risk)

        benchmark_metatdata_card_instances: list[BenchmarkMetadataCard] = (
            cls._risk_explorer.get_all_benchmark_metadata_cards(taxonomy)
        )
        return benchmark_metatdata_card_instances

    def get_benchmark_metadata_card(cls, id=str):
        """Get an benchmark_metadata_card definition from the LinkML, filtered by id

        Args:
            id: str
                The string id identifying the benchmark_metadata_card
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Action
                Result containing a benchmark_metadata_card.
        """
        type_check(
            "<RAN30946549E>",
            str,
            allow_none=False,
            id=id,
        )

        benchmark_metadata_card: BenchmarkMetadataCard | None = (
            cls._risk_explorer.get_benchmark_metadata_card(id=id)
        )
        return benchmark_metadata_card

    def get_documents(cls, taxonomy=None):
        """Get all document definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Documentation]
                Result containing a list of Documentation
        """
        type_check(
            "<RAN61770043E>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )

        document_instances: list[Documentation] = cls._risk_explorer.get_documents(
            taxonomy
        )
        return document_instances

    def get_document(cls, id=str):
        """Get a document definition from the LinkML, filtered by id

        Args:
            id: str
                The string id identifying the documentation entry
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Action
                Result containing a document.
        """
        type_check(
            "<RAN12472418E>",
            str,
            allow_none=False,
            id=id,
        )

        document: Documentation | None = cls._risk_explorer.get_document(id=id)
        return document

    def get_datasets(cls, taxonomy=None):
        """Get all dataset definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Dataset]
                Result containing a list of Dataset entries
        """
        type_check(
            "<RAN61770043E>",
            str,
            allow_none=True,
            taxonomy=taxonomy,
        )

        dataset_instances: list[Dataset] = cls._risk_explorer.get_datasets(taxonomy)
        return dataset_instances

    def get_dataset(cls, id=str):
        """Get a dataset definition from the LinkML, filtered by id

        Args:
            id: str
                The string id identifying the dataset entry
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Action
                Result containing a dataset.
        """
        type_check(
            "<RAN12472418E>",
            str,
            allow_none=False,
            id=id,
        )

        dataset: Dataset | None = cls._risk_explorer.get_dataset(id=id)
        return dataset

    def identify_domain_from_usecases(
        cls, usecases: List[str], inference_engine: InferenceEngine, verbose=True
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
        type_check(
            "<RAN4CDA6852E>",
            List,
            allow_none=False,
            usecases=usecases,
        )
        value_check(
            "<RAN0E435F50E>",
            inference_engine and usecases,
            "Please provide usecases and inference_engine",
        )

        # Load risk questionnaire CoT from the template dir
        risk_questionnaire = load_resource("risk_questionnaire_cot.json")

        # Retrieve domain question data
        domain_ques_data = risk_questionnaire[0]

        # Prepare few shots inference prompts from CoT Data
        prompts = [
            FewShotPromptBuilder(
                prompt_template=QUESTIONNAIRE_COT_TEMPLATE,
            ).build(
                cot_examples=domain_ques_data["cot_examples"],
                usecase=usecase,
                question=domain_ques_data["question"],
            )
            for usecase in usecases
        ]

        # Invoke inference service
        return inference_engine.chat(
            messages=prompts,
            response_format=DOMAIN_TYPE_SCHEMA,
            postprocessors=["json_object"],
            verbose=verbose,
        )

    def categorize_risk_severity(
        self,
        usecases: List[str],
        inference_engine: InferenceEngine,
    ):
        """Determine the severity of risks based on the use case description.
        Args:
            usecases (List[str]):
                A List of strings describing AI usecases
            inference_engine (InferenceEngine):
                An LLM inference engine
        Returns:
            results (List[Dict]):
                Results detailing risk categorization by usecase.
        """
        type_check(
            "<RAN3B9CD886E>",
            InferenceEngine,
            allow_none=False,
            inference_engine=inference_engine,
        )
        type_check(
            "<RAN4CDA6852E>",
            List,
            allow_none=False,
            usecases=usecases,
        )
        value_check(
            "<RAN0E435F50E>",
            inference_engine and usecases,
            "Please provide usecases and inference_engine",
        )

        # Create Risk Severity instance
        risk_severity = RiskSeverityCategorizer(inference_engine)

        # Load risk questionnaire from the template dir
        risk_questionnaire = load_resource("risk_questionnaire_cot.json")

        # Collecting required parameters for categorizing risk severity per usecase
        results = []
        for usecase in usecases:

            # Get AI Domain of the usecase
            domain = [
                domain.prediction["answer"]
                for domain in self.identify_domain_from_usecases(
                    [usecase], inference_engine=inference_engine, verbose=False
                )
            ][0]

            # Using a risk questionnaire to identify key attributes necessary for
            # constituting an AI system from the usecase.
            #
            # (Q4) AI User that interacts with the AI system
            # (Q5) Intended purpose of the AI System
            # (Q6) The capability of an AI system to do what it is designed to do
            # (Q7) AI Subject impacted by the AI System
            predictions = [
                response.prediction["answer"]
                for response in self.generate_few_shot_risk_questionnaire_output(
                    usecase,
                    list(
                        filter(
                            lambda question: question["no"] in ["Q4", "Q5", "Q6", "Q7"],
                            risk_questionnaire,
                        )
                    ),
                    inference_engine=inference_engine,
                    verbose=False,
                )
            ]

            # Extracting predictions
            ai_user = predictions[0]
            purpose = predictions[1]
            capability = predictions[2]
            ai_subject = predictions[3]

            # Calling the risk categorization API with the required attributes.
            results.append(
                risk_severity.categorize(
                    domain, purpose, capability, ai_user, ai_subject
                )
            )

        return results
