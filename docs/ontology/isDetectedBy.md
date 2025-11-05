

# Slot: isDetectedBy


_A relationship where a risk, risk source, consequence, or impact is detected by a risk control._





URI: [nexus:isDetectedBy](https://ibm.github.io/risk-atlas-nexus/ontology/isDetectedBy)
Alias: isDetectedBy

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |
| [RiskControl](RiskControl.md) | A measure that maintains and/or modifies risk (and risk concepts) |  no  |
| [RiskIncident](RiskIncident.md) | An event occuring or occured which is a realised or materialised risk |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |
| [Impact](Impact.md) |  |  no  |
| [RiskConcept](RiskConcept.md) | An umbrella term for refering to risk, risk source, consequence and impact |  no  |
| [RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |  no  |







## Properties

* Range: [RiskControl](RiskControl.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:isDetectedBy |
| native | nexus:isDetectedBy |




## LinkML Source

<details>
```yaml
name: isDetectedBy
description: A relationship where a risk, risk source, consequence, or impact is detected
  by a risk control.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
domain: RiskConcept
alias: isDetectedBy
domain_of:
- RiskConcept
inverse: detectsRiskConcept
range: RiskControl
multivalued: true
inlined: false

```
</details>
