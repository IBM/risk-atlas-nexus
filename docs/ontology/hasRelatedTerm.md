

# Slot: hasRelatedTerm


_A relationship where an entity relates to a term_





URI: [nexus:hasRelatedTerm](https://ibm.github.io/risk-atlas-nexus/ontology/hasRelatedTerm)
Alias: hasRelatedTerm

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LLMIntrinsic](LLMIntrinsic.md) | A capability that can be invoked through a well-defined API that is reasonabl... |  no  |







## Properties

* Range: [Term](Term.md)&nbsp;or&nbsp;<br />[RiskConcept](RiskConcept.md)&nbsp;or&nbsp;<br />[Term](Term.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasRelatedTerm |
| native | nexus:hasRelatedTerm |




## LinkML Source

<details>
```yaml
name: hasRelatedTerm
description: A relationship where an entity relates to a term
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
domain: Any
alias: hasRelatedTerm
domain_of:
- LLMIntrinsic
range: Term
multivalued: true
inlined: false
any_of:
- range: RiskConcept
- range: Term

```
</details>
