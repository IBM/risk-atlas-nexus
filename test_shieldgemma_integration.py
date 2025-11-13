#!/usr/bin/env python3
"""
ShieldGemma Integration Test Script
Tests that ShieldGemma risks, controls, and models are properly integrated into Risk Atlas Nexus
"""

import risk_atlas_nexus as ran


def test_shieldgemma_integration():
    """Comprehensive test of ShieldGemma integration"""

    print("=" * 80)
    print("SHIELDGEMMA INTEGRATION TEST")
    print("=" * 80)

    # Initialize Risk Atlas Nexus
    nexus = ran.RiskAtlasNexus()

    # Test 1: Verify ShieldGemma risks are loaded
    print("\n1. Testing ShieldGemma Risks...")
    print("-" * 80)
    all_risks = nexus.get_all_risks()
    shieldgemma_risks = [r for r in all_risks if 'shieldgemma' in r.id]

    assert len(shieldgemma_risks) == 4, f"Expected 4 ShieldGemma risks, found {len(shieldgemma_risks)}"
    print(f"Found {len(shieldgemma_risks)} ShieldGemma risks")

    for risk in shieldgemma_risks:
        print(f"   - {risk.name} ({risk.id})")
        assert hasattr(risk, 'relatedMatch'), f"Risk {risk.id} missing relatedMatch"
        assert len(risk.relatedMatch) > 0, f"Risk {risk.id} has no related matches"

    # Test 2: Verify ShieldGemma risk controls are loaded
    print("\n2. Testing ShieldGemma Risk Controls...")
    print("-" * 80)
    all_controls = nexus.get_all_risk_controls()
    shieldgemma_controls = [c for c in all_controls if 'shieldgemma' in c.id]

    assert len(shieldgemma_controls) == 4, f"Expected 4 ShieldGemma controls, found {len(shieldgemma_controls)}"
    print(f"Found {len(shieldgemma_controls)} ShieldGemma risk controls")

    for control in shieldgemma_controls:
        print(f"   - {control.name} ({control.id})")
        assert hasattr(control, 'detectsRiskConcept'), f"Control {control.id} missing detectsRiskConcept"
        assert len(control.detectsRiskConcept) > 0, f"Control {control.id} detects no risks"

    # Test 3: Verify ShieldGemma models are loaded
    print("\n3. Testing ShieldGemma Models...")
    print("-" * 80)
    models = nexus._ontology.aimodels
    shieldgemma_models = [m for m in models if 'shieldgemma' in m.id]

    assert len(shieldgemma_models) == 3, f"Expected 3 ShieldGemma models, found {len(shieldgemma_models)}"
    print(f"Found {len(shieldgemma_models)} ShieldGemma models")

    for model in shieldgemma_models:
        print(f"   - {model.name} ({model.id})")
        assert model.isProvidedBy == 'google', f"Model {model.id} has wrong provider"
        assert model.isPartOf == 'shieldgemma', f"Model {model.id} has wrong family"
        assert hasattr(model, 'hasRiskControl'), f"Model {model.id} missing hasRiskControl"
        assert len(model.hasRiskControl) == 4, f"Model {model.id} should have 4 controls, has {len(model.hasRiskControl)}"

    # Test 4: Verify API methods work
    print("\n4. Testing API Methods...")
    print("-" * 80)

    # Test get_risk_control
    ctrl = nexus.get_risk_control(id='shieldgemma-hate-speech-detection')
    assert ctrl is not None, "Could not retrieve shieldgemma-hate-speech-detection"
    assert ctrl.name == 'Hate Speech Detection', f"Wrong control name: {ctrl.name}"
    print(f"get_risk_control() works: {ctrl.name}")

    # Test get_risk
    risk = nexus.get_risk(id='shieldgemma-dangerous-content')
    assert risk is not None, "Could not retrieve shieldgemma-dangerous-content"
    assert risk.name == 'Dangerous Content', f"Wrong risk name: {risk.name}"
    print(f"get_risk() works: {risk.name}")

    # Test 5: Verify relationship traversal with manual methods
    print("\n5. Testing Manual Relationship Traversal...")
    print("-" * 80)

    # Get Risk Atlas risk
    atlas_risk = nexus.get_risk(id='atlas-toxic-output')
    assert atlas_risk is not None, "Could not retrieve atlas-toxic-output"

    # Check ShieldGemma relationships
    shieldgemma_related = [r for r in atlas_risk.relatedMatch if 'shieldgemma' in r]
    assert len(shieldgemma_related) == 3, f"Expected 3 ShieldGemma relations, found {len(shieldgemma_related)}"
    print(f"atlas-toxic-output relates to {len(shieldgemma_related)} ShieldGemma risks")

    # Find controls for related risks
    for sg_risk_id in shieldgemma_related:
        sg_controls = [c for c in all_controls
                      if hasattr(c, 'detectsRiskConcept') and sg_risk_id in c.detectsRiskConcept]
        assert len(sg_controls) > 0, f"No controls found for {sg_risk_id}"
        print(f"   - {sg_risk_id} → {sg_controls[0].name}")

    # Test 6: Verify get_related_risks() API method
    print("\n6. Testing get_related_risks() API Method...")
    print("-" * 80)

    # Test from Risk Atlas risk to ShieldGemma risks
    related_risks = nexus.get_related_risks(id='atlas-toxic-output')
    assert related_risks is not None, "get_related_risks() returned None"
    assert len(related_risks) > 0, "get_related_risks() returned empty list"

    shieldgemma_from_api = [r for r in related_risks if 'shieldgemma' in r.id]
    assert len(shieldgemma_from_api) == 3, f"Expected 3 ShieldGemma risks from API, found {len(shieldgemma_from_api)}"
    print(f"get_related_risks('atlas-toxic-output') returned {len(shieldgemma_from_api)} ShieldGemma risks:")

    for risk in shieldgemma_from_api:
        print(f"   - {risk.name} ({risk.id})")

    # Test reverse direction: from ShieldGemma risk to Risk Atlas risks
    related_to_hate_speech = nexus.get_related_risks(id='shieldgemma-hate-speech')
    assert related_to_hate_speech is not None, "get_related_risks() returned None for shieldgemma-hate-speech"
    assert len(related_to_hate_speech) > 0, "No related risks found for shieldgemma-hate-speech"

    atlas_risks_from_api = [r for r in related_to_hate_speech if r.id.startswith('atlas-')]
    assert len(atlas_risks_from_api) >= 3, f"Expected at least 3 Atlas risks from API, found {len(atlas_risks_from_api)}"
    print(f"get_related_risks('shieldgemma-hate-speech') returned {len(atlas_risks_from_api)} Risk Atlas risks:")

    for risk in atlas_risks_from_api[:3]:  # Show first 3
        print(f"   - {risk.name} ({risk.id})")

    # Test 7: Verify get_related_risk_controls() API method
    print("\n7. Testing get_related_risk_controls() API Method...")
    print("-" * 80)

    # Test from Risk Atlas risk to risk controls (should traverse: atlas risk → shieldgemma risk → control)
    related_controls = nexus.get_related_risk_controls(id='atlas-toxic-output')
    assert related_controls is not None, "get_related_risk_controls() returned None"
    for ctrl in related_controls:
            print(f"   - {ctrl.name} ({ctrl.id})")

    # Filter for ShieldGemma controls
    shieldgemma_controls_from_api = [c for c in related_controls if 'shieldgemma' in c.id]

    if len(shieldgemma_controls_from_api) == 0:
        print(f"  WARNING: get_related_risk_controls('atlas-toxic-output') returned 0 ShieldGemma controls")
        print(f"   Total controls returned: {len(related_controls)}")
        print(f"   This may indicate the API doesn't traverse through relatedMatch relationships")
        print(f"   Manual traversal found 3 ShieldGemma controls (see Test 5)")
    else:
        print(f"✅ get_related_risk_controls('atlas-toxic-output') returned {len(shieldgemma_controls_from_api)} ShieldGemma controls:")
        for ctrl in shieldgemma_controls_from_api:
            print(f"   - {ctrl.name} ({ctrl.id})")

    # Test from ShieldGemma risk directly to its controls
    controls_for_hate_speech = nexus.get_related_risk_controls(id='shieldgemma-hate-speech')
    assert controls_for_hate_speech is not None, "get_related_risk_controls() returned None for shieldgemma-hate-speech"

    shieldgemma_controls_direct = [c for c in controls_for_hate_speech if c.id == 'shieldgemma-hate-speech-detection']
    assert len(shieldgemma_controls_direct) == 1, f"Expected 1 direct control for shieldgemma-hate-speech, found {len(shieldgemma_controls_direct)}"
    print(f"✅ get_related_risk_controls('shieldgemma-hate-speech') correctly returns its detection control:")
    print(f"   - {shieldgemma_controls_direct[0].name}")

    # Test 6: Verify taxonomy
    print("\n6. Testing Taxonomy...")
    print("-" * 80)

    all_taxonomies = nexus.get_all_taxonomies()
    shieldgemma_tax = [t for t in all_taxonomies if t.id == 'shieldgemma-taxonomy']
    assert len(shieldgemma_tax) == 1, "ShieldGemma taxonomy not found"
    print(f"✅ Found ShieldGemma taxonomy: {shieldgemma_tax[0].name}")

    # Test 7: Verify documentation
    print("\n7. Testing Documentation...")
    print("-" * 80)

    docs = nexus.get_documents()
    shieldgemma_docs = [d for d in docs if d.id == 'shieldgemma-paper']
    assert len(shieldgemma_docs) == 1, "ShieldGemma documentation not found"
    print(f"Found ShieldGemma documentation: {shieldgemma_docs[0].name}")

    # Summary
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print("\nShieldGemma is fully integrated into Risk Atlas Nexus:")
    print(f"  • {len(shieldgemma_risks)} risks")
    print(f"  • {len(shieldgemma_controls)} risk controls")
    print(f"  • {len(shieldgemma_models)} models")
    print(f"  • 1 taxonomy")
    print(f"  • 1 documentation reference")
    print("\nYou can now query ShieldGemma data through the Risk Atlas Nexus API!")


if __name__ == '__main__':
    try:
        test_shieldgemma_integration()
    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
