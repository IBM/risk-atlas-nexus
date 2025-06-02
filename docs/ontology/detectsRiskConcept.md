

# Slot: detectsRiskConcept


_The property airo:detectsRiskConcept indicates the control used for detecting risks, risk sources,  consequences, and impacts._





URI: [nexus:detectsRiskConcept](https://ibm.github.io/risk-atlas-nexus/ontology/detectsRiskConcept)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |
| [RiskControl](RiskControl.md) | A measure that maintains and/or modifies risk (and risk concepts) |  no  |







## Properties

* Range: [RiskConcept](RiskConcept.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:detectsRiskConcept |
| native | nexus:detectsRiskConcept |
| exact | airo:detectsRiskConcept |




## LinkML Source

<details>
```yaml
name: detectsRiskConcept
description: The property airo:detectsRiskConcept indicates the control used for detecting
  risks, risk sources,  consequences, and impacts.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
exact_mappings:
- airo:detectsRiskConcept
rank: 1000
domain: RiskControl
alias: detectsRiskConcept
domain_of:
- Risk
- RiskControl
inverse: isDetectedBy
range: RiskConcept
multivalued: true
inlined: false

```
</details>
