

# Slot: hasLicense


_Indicates licenses associated with a resource_





URI: [airo:hasLicense](https://w3id.org/airo#hasLicense)
Alias: hasLicense

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc |  no  |
| [Vocabulary](Vocabulary.md) | A collection of terms, with their definitions and relationships |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |  no  |
| [RiskTaxonomy](RiskTaxonomy.md) | A taxonomy of AI system related risks |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |
| [Documentation](Documentation.md) | Documented information about a concept or other topic(s) of interest |  no  |
| [Dataset](Dataset.md) | A body of structured information describing some topic(s) of interest |  no  |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is c... |  no  |







## Properties

* Range: [License](License.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:hasLicense |
| native | nexus:hasLicense |




## LinkML Source

<details>
```yaml
name: hasLicense
description: Indicates licenses associated with a resource
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: airo:hasLicense
alias: hasLicense
domain_of:
- Dataset
- Documentation
- Vocabulary
- RiskTaxonomy
- BaseAi
- AiEval
- BenchmarkMetadataCard
- Adapter
range: License

```
</details>
