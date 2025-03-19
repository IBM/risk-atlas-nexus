

# Slot: hasModelCard


_A relationship to model card references._





URI: [nexus:hasModelCard](http://research.ibm.com/ontologies/aiont/hasModelCard)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is c... |  no  |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |
| [BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc |  no  |







## Properties

* Range: [String](String.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasModelCard |
| native | nexus:hasModelCard |




## LinkML Source

<details>
```yaml
name: hasModelCard
description: A relationship to model card references.
from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
rank: 1000
alias: hasModelCard
domain_of:
- BaseAi
range: string
multivalued: true
inlined: true
inlined_as_list: true

```
</details>