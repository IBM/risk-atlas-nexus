

# Slot: isDetectedBy


_A relationship where a risk, risk source, consequence, or impact is detected by a risk control._





URI: [nexus:isDetectedBy](https://ibm.github.io/risk-atlas-nexus/ontology/isDetectedBy)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RiskConcept](RiskConcept.md) | An umbrella term for refering to risk, risk source, consequence and impact |  no  |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |
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
alias: isDetectedBy
domain_of:
- RiskConcept
inverse: detectsRiskConcept
range: RiskControl
multivalued: true
inlined: false

```
</details>