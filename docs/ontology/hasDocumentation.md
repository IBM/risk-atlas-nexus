

# Slot: hasDocumentation


_Indicates documentation associated with an entity._





URI: [airo:hasDocumentation](https://w3id.org/airo#hasDocumentation)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModelFamily](LargeLanguageModelFamily.md) | A large language model family is a set of models that are provided by the sam... |  no  |
| [BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |
| [RiskTaxonomy](RiskTaxonomy.md) | A taxonomy of AI system related risks |  no  |
| [AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is c... |  no  |
| [AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities |  no  |
| [Dataset](Dataset.md) | A body of structured information describing some topic(s) of interest |  no  |







## Properties

* Range: [Documentation](Documentation.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:hasDocumentation |
| native | nexus:hasDocumentation |




## LinkML Source

<details>
```yaml
name: hasDocumentation
description: Indicates documentation associated with an entity.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: airo:hasDocumentation
alias: hasDocumentation
domain_of:
- Dataset
- RiskTaxonomy
- Action
- AiEval
- BaseAi
- LargeLanguageModelFamily
range: Documentation
multivalued: true
inlined: false

```
</details>