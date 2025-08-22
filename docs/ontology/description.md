

# Slot: description


_The description of an entity_





URI: [schema:description](http://schema.org/description)
Alias: description

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [IncidentOngoingclass](IncidentOngoingclass.md) |  |  no  |
| [IncidentMitigatedclass](IncidentMitigatedclass.md) |  |  no  |
| [RiskConcept](RiskConcept.md) | An umbrella term for refering to risk, risk source, consequence and impact |  no  |
| [AiModelValidation](AiModelValidation.md) | AI model validation steps that have been performed after the model training t... |  no  |
| [Organization](Organization.md) | Any organizational entity such as a corporation, educational institution, con... |  no  |
| [Term](Term.md) | A term and its definitions |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |
| [LLMIntrinsic](LLMIntrinsic.md) | A capability that can be invoked through a well-defined API that is reasonabl... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |
| [Input](Input.md) | Input for which the system or component generates output |  no  |
| [AiProvider](AiProvider.md) | A provider under the AI Act is defined by Article 3(3) as a natural or legal ... |  no  |
| [AiEvalResult](AiEvalResult.md) | The result of an evaluation for a specific AI model |  no  |
| [RiskIncident](RiskIncident.md) | An event occuring or occured which is a realised or materialised risk |  no  |
| [AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities |  no  |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [AiLifecyclePhase](AiLifecyclePhase.md) | A Phase of AI lifecycle which indicates evolution of the system from concepti... |  no  |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |  no  |
| [Dataset](Dataset.md) | A body of structured information describing some topic(s) of interest |  no  |
| [AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is c... |  no  |
| [AiTask](AiTask.md) | A task, such as summarization and classification, performed by an AI |  no  |
| [Consequence](Consequence.md) |  |  no  |
| [BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc |  no  |
| [RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |  no  |
| [IncidentConcludedclass](IncidentConcludedclass.md) |  |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [AiOffice](AiOffice.md) | The EU AI Office (https://digital-strategy |  no  |
| [IncidentHaltedclass](IncidentHaltedclass.md) |  |  no  |
| [IncidentStatus](IncidentStatus.md) |  |  no  |
| [IncidentNearMissclass](IncidentNearMissclass.md) |  |  no  |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |
| [License](License.md) | The general notion of a license which defines terms and grants permissions to... |  no  |
| [Likelihood](Likelihood.md) |  |  no  |
| [Documentation](Documentation.md) | Documented information about a concept or other topic(s) of interest |  no  |
| [DataPreprocessing](DataPreprocessing.md) | Data transformations, such as PI filtering, performed to ensure high quality ... |  no  |
| [Impact](Impact.md) |  |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [Modality](Modality.md) | A modality supported by an Ai component |  no  |
| [Entity](Entity.md) | A generic grouping for any identifiable entity |  no  |
| [RiskTaxonomy](RiskTaxonomy.md) | A taxonomy of AI system related risks |  no  |
| [LargeLanguageModelFamily](LargeLanguageModelFamily.md) | A large language model family is a set of models that are provided by the sam... |  no  |
| [RiskControl](RiskControl.md) | A measure that maintains and/or modifies risk (and risk concepts) |  no  |
| [Vocabulary](Vocabulary.md) | A collection of terms, with their definitions and relationships |  no  |
| [Severity](Severity.md) |  |  no  |







## Properties

* Range: [String](String.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:description |
| native | nexus:description |




## LinkML Source

<details>
```yaml
name: description
description: The description of an entity
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: schema:description
alias: description
domain_of:
- Entity
range: string

```
</details>
