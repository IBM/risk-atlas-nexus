

# Slot: hasRule


_Specifying applicability or inclusion of a rule within specified context._





URI: [dpv:hasRule](https://w3id.org/dpv#hasRule)
Alias: hasRule

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LLMQuestionPolicy](LLMQuestionPolicy.md) | The policy guides how the language model should answer a diverse set of sensi... |  no  |







## Properties

* Range: [Rule](Rule.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | dpv:hasRule |
| native | nexus:hasRule |




## LinkML Source

<details>
```yaml
name: hasRule
description: Specifying applicability or inclusion of a rule within specified context.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: dpv:hasRule
alias: hasRule
domain_of:
- LLMQuestionPolicy
range: Rule
multivalued: true
inlined: false

```
</details>
