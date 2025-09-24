

# Slot: exactMatch


_The property is used to link two concepts, indicating a high degree of confidence that the concepts can be used interchangeably across a wide range of information retrieval applications_





URI: [skos:exactMatch](http://www.w3.org/2004/02/skos/core/exactMatch)
Alias: exactMatch

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |  no  |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |







## Properties

* Range: [Any](Any.md)&nbsp;or&nbsp;<br />[Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskGroup](RiskGroup.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | skos:exactMatch |
| native | nexus:exactMatch |




## LinkML Source

<details>
```yaml
name: exactMatch
description: The property is used to link two concepts, indicating a high degree of
  confidence that the concepts can be used interchangeably across a wide range of
  information retrieval applications
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: skos:exactMatch
alias: exactMatch
domain_of:
- RiskGroup
- Risk
range: Any
multivalued: true
inlined: false
any_of:
- range: Risk
- range: RiskGroup

```
</details>
