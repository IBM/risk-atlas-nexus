

# Slot: hasDocumentation


_Indicates documentation associated with an entity._





URI: [airo:hasDocumentation](https://w3id.org/airo#hasDocumentation)
Alias: hasDocumentation

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc |  no  |
| [LargeLanguageModelFamily](LargeLanguageModelFamily.md) | A large language model family is a set of models that are provided by the sam... |  no  |
| [Vocabulary](Vocabulary.md) | A collection of terms, with their definitions and relationships |  no  |
| [Principle](Principle.md) | A representation of values or norms that must be taken into consideration whe... |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |
| [Term](Term.md) | A term and its definitions |  no  |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |  no  |
| [RiskTaxonomy](RiskTaxonomy.md) | A taxonomy of AI system related risks |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |
| [Dataset](Dataset.md) | A body of structured information describing some topic(s) of interest |  no  |
| [LLMIntrinsic](LLMIntrinsic.md) | A capability that can be invoked through a well-defined API that is reasonabl... |  no  |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is c... |  no  |







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
- Vocabulary
- Term
- Principle
- RiskTaxonomy
- Action
- BaseAi
- LargeLanguageModelFamily
- AiEval
- BenchmarkMetadataCard
- Adapter
- LLMIntrinsic
range: Documentation
multivalued: true
inlined: false

```
</details>
