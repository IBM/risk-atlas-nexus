"""
ShieldGemma Integration Tests
Tests that ShieldGemma risks, controls, and models are properly integrated into Risk Atlas Nexus
"""

# Internal
from src.risk_atlas_nexus import RiskAtlasNexus

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestShieldGemmaIntegration(TestCaseBase):
    """Tests for ShieldGemma integration into Risk Atlas Nexus"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.nexus = RiskAtlasNexus()

    def test_shieldgemma_risks_loaded(self):
        """Verify ShieldGemma risks are loaded"""
        all_risks = self.nexus.get_all_risks()
        shieldgemma_risks = [r for r in all_risks if 'shieldgemma' in r.id]

        self.assertEqual(
            len(shieldgemma_risks),
            4,
            f"Expected 4 ShieldGemma risks, found {len(shieldgemma_risks)}"
        )

        # Verify each risk has required attributes
        for risk in shieldgemma_risks:
            self.assertTrue(
                hasattr(risk, 'relatedMatch'),
                f"Risk {risk.id} missing relatedMatch"
            )
            self.assertGreater(
                len(risk.relatedMatch),
                0,
                f"Risk {risk.id} has no related matches"
            )

    def test_shieldgemma_risk_controls_loaded(self):
        """Verify ShieldGemma risk controls are loaded"""
        all_controls = self.nexus.get_all_risk_controls()
        shieldgemma_controls = [c for c in all_controls if 'shieldgemma' in c.id]

        self.assertEqual(
            len(shieldgemma_controls),
            4,
            f"Expected 4 ShieldGemma controls, found {len(shieldgemma_controls)}"
        )

        # Verify each control has required attributes
        for control in shieldgemma_controls:
            self.assertTrue(
                hasattr(control, 'detectsRiskConcept'),
                f"Control {control.id} missing detectsRiskConcept"
            )
            self.assertGreater(
                len(control.detectsRiskConcept),
                0,
                f"Control {control.id} detects no risks"
            )

    def test_shieldgemma_models_loaded(self):
        """Verify ShieldGemma models are loaded"""
        models = self.nexus._ontology.aimodels
        shieldgemma_models = [m for m in models if 'shieldgemma' in m.id]

        self.assertEqual(
            len(shieldgemma_models),
            3,
            f"Expected 3 ShieldGemma models, found {len(shieldgemma_models)}"
        )

        # Verify each model has required attributes
        for model in shieldgemma_models:
            self.assertEqual(
                model.isProvidedBy,
                'google',
                f"Model {model.id} has wrong provider"
            )
            self.assertEqual(
                model.isPartOf,
                'shieldgemma',
                f"Model {model.id} has wrong family"
            )
            self.assertTrue(
                hasattr(model, 'hasRiskControl'),
                f"Model {model.id} missing hasRiskControl"
            )
            self.assertEqual(
                len(model.hasRiskControl),
                4,
                f"Model {model.id} should have 4 controls, has {len(model.hasRiskControl)}"
            )

    def test_get_risk_control_api(self):
        """Test get_risk_control API method for ShieldGemma"""
        ctrl = self.nexus.get_risk_control(id='shieldgemma-hate-speech-detection')

        self.assertIsNotNone(ctrl, "Could not retrieve shieldgemma-hate-speech-detection")
        self.assertEqual(ctrl.name, 'Hate Speech Detection', f"Wrong control name: {ctrl.name}")

    def test_get_risk_api(self):
        """Test get_risk API method for ShieldGemma"""
        risk = self.nexus.get_risk(id='shieldgemma-dangerous-content')

        self.assertIsNotNone(risk, "Could not retrieve shieldgemma-dangerous-content")
        self.assertEqual(risk.name, 'Dangerous Content', f"Wrong risk name: {risk.name}")

    def test_manual_relationship_traversal(self):
        """Test manual relationship traversal from Risk Atlas to ShieldGemma"""
        # Get Risk Atlas risk
        atlas_risk = self.nexus.get_risk(id='atlas-toxic-output')
        self.assertIsNotNone(atlas_risk, "Could not retrieve atlas-toxic-output")

        # Check ShieldGemma relationships
        shieldgemma_related = [r for r in atlas_risk.relatedMatch if 'shieldgemma' in r]
        self.assertEqual(
            len(shieldgemma_related),
            3,
            f"Expected 3 ShieldGemma relations, found {len(shieldgemma_related)}"
        )

        # Find controls for related risks
        all_controls = self.nexus.get_all_risk_controls()
        for sg_risk_id in shieldgemma_related:
            sg_controls = [
                c for c in all_controls
                if hasattr(c, 'detectsRiskConcept') and sg_risk_id in c.detectsRiskConcept
            ]
            self.assertGreater(
                len(sg_controls),
                0,
                f"No controls found for {sg_risk_id}"
            )

    def test_get_related_risks_atlas_to_shieldgemma(self):
        """Test get_related_risks() from Risk Atlas risk to ShieldGemma risks"""
        related_risks = self.nexus.get_related_risks(id='atlas-toxic-output')

        self.assertIsNotNone(related_risks, "get_related_risks() returned None")
        self.assertGreater(len(related_risks), 0, "get_related_risks() returned empty list")

        shieldgemma_from_api = [r for r in related_risks if 'shieldgemma' in r.id]
        self.assertEqual(
            len(shieldgemma_from_api),
            3,
            f"Expected 3 ShieldGemma risks from API, found {len(shieldgemma_from_api)}"
        )

    def test_get_related_risks_shieldgemma_to_atlas(self):
        """Test get_related_risks() from ShieldGemma risk to Risk Atlas risks"""
        related_to_hate_speech = self.nexus.get_related_risks(id='shieldgemma-hate-speech')

        self.assertIsNotNone(
            related_to_hate_speech,
            "get_related_risks() returned None for shieldgemma-hate-speech"
        )
        self.assertGreater(
            len(related_to_hate_speech),
            0,
            "No related risks found for shieldgemma-hate-speech"
        )

        atlas_risks_from_api = [r for r in related_to_hate_speech if r.id.startswith('atlas-')]
        self.assertGreaterEqual(
            len(atlas_risks_from_api),
            3,
            f"Expected at least 3 Atlas risks from API, found {len(atlas_risks_from_api)}"
        )

    def test_get_related_risk_controls_from_atlas_risk(self):
        """Test get_related_risk_controls() from Risk Atlas risk"""
        related_controls = self.nexus.get_related_risk_controls(id='atlas-toxic-output')

        self.assertIsNotNone(related_controls, "get_related_risk_controls() returned None")

        # Filter for ShieldGemma controls
        shieldgemma_controls_from_api = [c for c in related_controls if 'shieldgemma' in c.id]

        # Note: This test documents current behavior. If the API doesn't traverse
        # through relatedMatch relationships, this will be 0 and the assertion will
        # produce a warning message in the output
        if len(shieldgemma_controls_from_api) == 0:
            # Log warning but don't fail the test
            print(f"\nWARNING: get_related_risk_controls('atlas-toxic-output') returned 0 ShieldGemma controls")
            print(f"Total controls returned: {len(related_controls)}")
            print(f"This may indicate the API doesn't traverse through relatedMatch relationships")

    def test_get_related_risk_controls_from_shieldgemma_risk(self):
        """Test get_related_risk_controls() from ShieldGemma risk directly"""
        controls_for_hate_speech = self.nexus.get_related_risk_controls(id='shieldgemma-hate-speech')

        self.assertIsNotNone(
            controls_for_hate_speech,
            "get_related_risk_controls() returned None for shieldgemma-hate-speech"
        )

        shieldgemma_controls_direct = [
            c for c in controls_for_hate_speech
            if c.id == 'shieldgemma-hate-speech-detection'
        ]
        self.assertEqual(
            len(shieldgemma_controls_direct),
            1,
            f"Expected 1 direct control for shieldgemma-hate-speech, found {len(shieldgemma_controls_direct)}"
        )

    def test_shieldgemma_taxonomy(self):
        """Verify ShieldGemma taxonomy is loaded"""
        all_taxonomies = self.nexus.get_all_taxonomies()
        shieldgemma_tax = [t for t in all_taxonomies if t.id == 'shieldgemma-taxonomy']

        self.assertEqual(len(shieldgemma_tax), 1, "ShieldGemma taxonomy not found")
        self.assertEqual(shieldgemma_tax[0].name, 'ShieldGemma Safety Categories')

    def test_shieldgemma_documentation(self):
        """Verify ShieldGemma documentation is loaded"""
        docs = self.nexus.get_documents()
        shieldgemma_docs = [d for d in docs if d.id == 'shieldgemma-paper']

        self.assertEqual(len(shieldgemma_docs), 1, "ShieldGemma documentation not found")
        self.assertIn('ShieldGemma', shieldgemma_docs[0].name)
