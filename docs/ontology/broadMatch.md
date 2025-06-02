

# Slot: broadMatch


_The property is used to state a hierarchical mapping link between two concepts, indicating that the concept linked to, is a broader concept than the originating concept._





URI: [skos:broadMatch](http://www.w3.org/2004/02/skos/core/broadMatch)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |
| [RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |  no  |







## Properties

* Range: [Any](Any.md)&nbsp;or&nbsp;<br />[Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskGroup](RiskGroup.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | skos:broadMatch |
| native | nexus:broadMatch |




## LinkML Source

<details>
```yaml
name: broadMatch
description: The property is used to state a hierarchical mapping link between two
  concepts, indicating that the concept linked to, is a broader concept than the originating
  concept.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: skos:broadMatch
alias: broadMatch
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
