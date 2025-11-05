# Standard
import os
import tempfile
from typing import Dict, List, Union

# Third party
from linkml_runtime import SchemaView
from linkml_runtime.dumpers import YAMLDumper
from sssom_schema import Mapping

# Internal
from src.risk_atlas_nexus import LLMIntrinsic, RiskAtlasNexus

# Unit Test Infrastructure
from src.risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import (
    Action,
    AiEval,
    BenchmarkMetadataCard,
    Dataset,
    Documentation,
    Principle,
    Risk,
    RiskIncident,
    Stakeholder,
)
from tests.base import TestCaseBase


class TestLibrary(TestCaseBase):
    """Tests for Library"""

    @classmethod
    def setUpClass(cls):
        cls.data_dir_path = "some_path"
        ran_lib = RiskAtlasNexus()
        cls.ran_lib = ran_lib

    @classmethod
    def tearDownClass(cls):
        return NotImplemented


    ###############################################################################################
    #                                 Tests for .get_schema()                                    #
    ###############################################################################################
    def test_get_schema(self):
        schema = self.ran_lib.get_schema()
        assert isinstance(schema, SchemaView)

    ###############################################################################################
    #                                 Tests for .export()                                         #
    ###############################################################################################

    def test_export(self):
        """Test RiskAtlasNexus can save"""
        with tempfile.TemporaryDirectory() as tempdir:
            ran_lib = self.ran_lib
            ran_lib.export(tempdir)
            self.assertTrue(
                os.path.exists(os.path.join(tempdir, "ai-risk-ontology.yaml"))
            )

    ###############################################################################################
    #                                 Tests for risks exploration                               #
    ###############################################################################################

    def test_own_base_dir_actually_exists(self):
        self.assertRaises(FileNotFoundError, RiskAtlasNexus, "/an_unlikely_file_path")

    def test_get_all_risks_type(self):
        """Check type of Get all risk definitions"""
        ran_lib = self.ran_lib
        all_risks = ran_lib.get_all_risks()
        self.assertIsInstance(all_risks, list)

    def test_get_all_risks_type_of_list_item(self):
        """Check type of Get all risk definitions, inner item"""
        ran_lib = self.ran_lib
        all_risks = ran_lib.get_all_risks()
        assert all_risks[0].linkml_meta.root["class_uri"] == "airo:Risk"

    def test_get_risk_by_tag_type(self):
        """Check type of Get risk definition filtered by risk atlas tag"""
        ran_lib = self.ran_lib
        risk = ran_lib.get_risk(tag="toxic-output")
        assert risk.linkml_meta.root["class_uri"] == "airo:Risk"

    def test_get_risk_by_tag_match(self):
        """Check content of Get risk definition filtered by risk atlas tag"""
        ran_lib = self.ran_lib
        risk = ran_lib.get_risk(tag="toxic-output")
        assert risk.tag == "toxic-output"

    def test_get_risk_by_id_type(self):
        """Check type of Get risk definition filtered by risk ID"""
        ran_lib = self.ran_lib
        risk = ran_lib.get_risk(id="atlas-toxic-output")
        assert risk.linkml_meta.root["class_uri"] == "airo:Risk"

    def test_get_related_risks_by_tag(self):
        """Get related risk definitions from the LinkML, by risk atlas tag"""
        ran_lib = self.ran_lib
        risks = ran_lib.get_related_risks(tag="toxic-output")
        assert all(i.linkml_meta.root["class_uri"] == "airo:Risk" for i in risks)
        self.assertGreater(len(risks), 0)

    def test_get_related_risk_ids_by_tag(self):
        """Get related risk definitions from the LinkML, by risk atlas tag"""
        ran_lib = self.ran_lib
        risks = ran_lib.get_related_risks(tag="toxic-output")
        assert all(i.linkml_meta.root["class_uri"] == "airo:Risk" for i in risks)
        self.assertGreater(len(risks), 0)

    def test_get_related_risks_by_id(self):
        """Get related risk definitions from the LinkML, by risk id"""
        ran_lib = self.ran_lib
        risks = ran_lib.get_related_risks(id="granite-function-call")
        assert all(i.linkml_meta.root["class_uri"] == "airo:Risk" for i in risks)
        self.assertGreater(len(risks), 0)

    def test_get_related_risk_ids_by_id(self):
        """Get related risk definitions from the LinkML, by risk id"""
        ran_lib = self.ran_lib
        risks = ran_lib.get_related_risks(id="granite-function-call")
        self.assertGreater(len(risks), 0)

    def test_get_risk_actions_by_risk_id(self):
        """Get related actions definitions from the LinkML, by risk id"""
        ran_lib = self.ran_lib
        risks = ran_lib.get_related_actions(id="nist-human-ai-configuration")
        self.assertGreater(len(risks), 0)

    def test_get_all_actions_type(self):
        """Check type of Get all action definitions"""
        ran_lib = self.ran_lib
        all_actions = ran_lib.get_all_actions()
        self.assertIsInstance(all_actions, list)

    def test_identify_risks_from_usecase_taxonomy_not_a_str(self):
        """Identify potential risks from a usecase description - taxonomy type is wrong"""
        ran_lib = self.ran_lib
        self.assertRaises(
            TypeError,
            ran_lib.identify_risks_from_usecases,
            "my usecase",
            "my inference_engine",
            4000,
        )

    def test_identify_risks_from_usecase_wrong_taxonomy(self):
        """Identify potential risks from a usecase description - taxonomy not found"""
        ran_lib = self.ran_lib
        self.assertRaises(
            Exception,
            ran_lib.identify_risks_from_usecases,
            "my usecase",
            "my inference_engine",
            "non-existing-taxonomy",
        )

    def test_get_all_taxonomies(self):
        """Get all taxonomy definitions from the LinkML"""
        ran_lib = self.ran_lib
        taxonomies = ran_lib.get_all_taxonomies()
        self.assertGreater(len(taxonomies), 0)

    def test_get_taxonomy_by_id(self):
        """Get taxonomy definitions from the LinkML filtered by taxonomy id"""
        ran_lib = self.ran_lib
        taxonomy = ran_lib.get_taxonomy_by_id("nist-ai-rmf")
        assert taxonomy.id == "nist-ai-rmf"

    def test_get_all_risk_controls(self):
        """Get all risk control definitions from the LinkML"""
        ran_lib = self.ran_lib
        risk_controls = ran_lib.get_all_risk_controls()
        self.assertGreater(len(risk_controls), 0)

    def test_get_risk_control_by_id(self):
        """Get risk_control definition from the LinkML filtered by risk_control id"""
        ran_lib = self.ran_lib
        risk_control = ran_lib.get_risk_control(id="gg-unethical-behavior-detection")
        assert risk_control.id == "gg-unethical-behavior-detection"

    def test_generate_proposed_mappings(self):
        """Test Identify mappings between a new set of risks and risks that exist in the Risk Atlas"""
        ran_lib = self.ran_lib

        risks_1 = ran_lib.get_all_risks(taxonomy="nist-ai-rmf")
        risks_2 = ran_lib.get_all_risks(taxonomy="ibm-risk-atlas")

        mappings = ran_lib.generate_proposed_mappings(
            risks_1, risks_2, None, "test_prefix", "SEMANTIC"
        )

        self.assertGreater(len(mappings), 0)
        self.assertIsInstance(mappings, List)
        assert all(isinstance(i, Mapping) for i in mappings)

        with self.assertRaises(ValueError):
            mappings2 = ran_lib.generate_proposed_mappings(
                risks_1, None, None, "test_prefix", "SEMANTIC"
            )

    def test_get_risk_incidents(self):
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._riskincidents = [RiskIncident(id="test-ri")]
        ris = ran_lib.get_risk_incidents()
        assert all(isinstance(i, RiskIncident) for i in ris)
        self.assertIs(len(ris), 1)

    def test_get_risk_incident(self):
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._riskincidents = [RiskIncident(id="test-ri")]
        ri = ran_lib.get_risk_incident(id="test-ri")
        assert isinstance(ri, RiskIncident)

    def test_get_related_risk_incidents(self):
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._riskincidents = [
            RiskIncident(id="test-ri", refersToRisk=["atlas-data-bias"])
        ]
        ri = ran_lib.get_risk_incident(id="test-ri")
        assert isinstance(ri, RiskIncident)
        rris = ran_lib.get_related_risk_incidents(risk_id="atlas-data-bias")
        assert all(isinstance(i, RiskIncident) for i in rris)
        self.assertIs(len(rris), 1)
        incident = rris[0]
        self.assertEqual(incident.id, "test-ri")
        self.assertIn("atlas-data-bias", incident.refersToRisk)

    def test_get_all_evaluations(self):
        """Get all evaluation definitions from the LinkML"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._evaluations = [AiEval(id="test-eval1")]
        evaluations = ran_lib.get_all_evaluations()
        self.assertGreater(len(evaluations), 0)

    def test_get_evaluation_by_id(self):
        """Get evaluation definition from the LinkML filtered by evaluation id"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._evaluations = [AiEval(id="test-eval1")]
        evaluation = ran_lib.get_evaluation(id="test-eval1")
        assert evaluation.id == "test-eval1"

    def test_get_related_evaluations(self):
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._evaluations = ran_lib._risk_explorer._evaluations + [
            AiEval(id="test-eval1", hasRelatedRisk=["atlas-data-bias"])
        ]
        ev1 = ran_lib.get_evaluation(id="test-eval1")
        assert isinstance(ev1, AiEval)
        evs = ran_lib.get_related_evaluations(risk_id="atlas-data-bias")
        assert all(isinstance(i, AiEval) for i in evs)
        self.assertIs(len(evs), 1)
        ev = evs[0]
        self.assertEqual(ev.id, "test-eval1")
        self.assertIn("atlas-data-bias", ev.hasRelatedRisk)

    def test_get_all_benchmark_metadata_cards(self):
        """Get all benchmark_metadata_card definitions from the LinkML"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._benchmarkmetadatacards = [
            BenchmarkMetadataCard(id="test-bmc1")
        ]
        bmcs = ran_lib.get_benchmark_metadata_cards()
        self.assertGreater(len(bmcs), 0)

    def test_get_benchmark_metadata_card_by_id(self):
        """Get benchmark metadata card definition from the LinkML filtered by dataset id"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._benchmarkmetadatacards = [
            BenchmarkMetadataCard(id="test-bmc1")
        ]
        bmc = ran_lib.get_benchmark_metadata_card(id="test-bmc1")
        assert bmc.id == "test-bmc1"

    def test_get_all_documents(self):
        """Get all documents definitions from the LinkML"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._documents = [Documentation(id="test-ds1")]
        documents = ran_lib.get_documents()
        self.assertGreater(len(documents), 0)

    def test_get_document_by_id(self):
        """Get document definition from the LinkML filtered by dataset id"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._documents = [Documentation(id="test-ds1")]
        document = ran_lib.get_document(id="test-ds1")
        assert document.id == "test-ds1"

    def test_get_all_datasets(self):
        """Get all dataset definitions from the LinkML"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._datasets = [Dataset(id="test-ds1")]
        datasets = ran_lib.get_datasets()
        self.assertGreater(len(datasets), 0)

    def test_get_dataset_by_id(self):
        """Get dataset definition from the LinkML filtered by dataset id"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._datasets = [Dataset(id="test-ds1")]
        dataset = ran_lib.get_dataset(id="test-ds1")
        assert dataset.id == "test-ds1"

    def test_get_all_stakeholders(self):
        """Get all stakeholder definitions from the LinkML"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._stakeholders = [Stakeholder(id="test-stakeholder1")]
        stakeholders = ran_lib.get_stakeholders()
        self.assertGreater(len(stakeholders), 0)

    def test_get_document_by_id(self):
        """Get stakeholder definition from the LinkML filtered by stakeholder id"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._stakeholders = [Documentation(id="test-stakeholder1")]
        stakeholder = ran_lib.get_stakeholder(id="test-stakeholder1")
        assert stakeholder.id == "test-stakeholder1"

    def test_get_all_intrinsics(self):
        """Get all llm intrinsic definitions from the LinkML"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._llmintrinsics = [LLMIntrinsic(id="test-intrinsic")]
        llm_intrinsics = ran_lib.get_intrinsics()
        self.assertGreater(len(llm_intrinsics), 0)

    def test_get_intrinsic_by_id(self):
        """Get intrinsic definition from the LinkML filtered by intrinsic id"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._llmintrinsics= [LLMIntrinsic(id="test-intrinsic1")]
        intrinsic = ran_lib.get_intrinsic(id="test-intrinsic1")
        assert intrinsic.id == "test-intrinsic1"

    def test_get_related_intrinsics(self):
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._llmintrinsics = ran_lib._risk_explorer._llmintrinsics + [
            LLMIntrinsic(id="test-intrinsic1", hasRelatedRisk=["atlas-data-bias"])
        ]
        ev1 = ran_lib.get_intrinsic(id="test-intrinsic1")
        assert isinstance(ev1, LLMIntrinsic)
        evs = ran_lib.get_related_intrinsics(risk_id="atlas-data-bias")
        assert all(isinstance(i, LLMIntrinsic) for i in evs)

    def test_get_all_principles(self):
        """Get all principle definitions from the LinkML"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._principles = [Principle(id="test-principle")]
        principles = ran_lib.get_principles()
        self.assertGreater(len(principles), 0)

    def test_get_principle_by_id(self):
        """Get principle definition from the LinkML filtered by principle id"""
        ran_lib = self.ran_lib
        ran_lib._risk_explorer._principles = [Principle(id="test-principle1")]
        principle = ran_lib.get_principle(id="test-principle1")
        assert principle.id == "test-principle1"
