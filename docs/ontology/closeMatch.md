

# Slot: closeMatch


_The property is used to link two concepts that are sufficiently similar that they can be used interchangeably in some information retrieval applications._





URI: [skos:closeMatch](http://www.w3.org/2004/02/skos/core/closeMatch)
Alias: closeMatch

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
| self | skos:closeMatch |
| native | nexus:closeMatch |




## LinkML Source

<details>
```yaml
name: closeMatch
description: The property is used to link two concepts that are sufficiently similar
  that they can be used interchangeably in some information retrieval applications.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: skos:closeMatch
alias: closeMatch
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
