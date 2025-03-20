

# Slot: performsTask


_relationship indicating the AI tasks an AI model can perform._





URI: [nexus:performsTask](http://research.ibm.com/ontologies/aiont/performsTask)



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

* Range: [AiTask](AiTask.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:performsTask |
| native | nexus:performsTask |




## LinkML Source

<details>
```yaml
name: performsTask
description: relationship indicating the AI tasks an AI model can perform.
from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
rank: 1000
alias: performsTask
domain_of:
- BaseAi
range: AiTask
multivalued: true
inlined: false

```
</details>