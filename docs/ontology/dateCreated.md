

# Slot: dateCreated


_The date on which the entity was created._





URI: [schema:dateCreated](http://schema.org/dateCreated)
Alias: dateCreated

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LLMIntrinsic](LLMIntrinsic.md) | A capability that can be invoked through a well-defined API that is reasonabl... |  no  |
| [Severity](Severity.md) |  |  no  |
| [Entity](Entity.md) | A generic grouping for any identifiable entity |  no  |
| [Documentation](Documentation.md) | Documented information about a concept or other topic(s) of interest |  no  |
| [Impact](Impact.md) |  |  no  |
| [Input](Input.md) | Input for which the system or component generates output |  no  |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |  no  |
| [License](License.md) | The general notion of a license which defines terms and grants permissions to... |  no  |
| [AiModelValidation](AiModelValidation.md) | AI model validation steps that have been performed after the model training t... |  no  |
| [Consequence](Consequence.md) |  |  no  |
| [IncidentMitigatedclass](IncidentMitigatedclass.md) |  |  no  |
| [IncidentOngoingclass](IncidentOngoingclass.md) |  |  no  |
| [AiOffice](AiOffice.md) | The EU AI Office (https://digital-strategy |  no  |
| [AiLifecyclePhase](AiLifecyclePhase.md) | A Phase of AI lifecycle which indicates evolution of the system from concepti... |  no  |
| [BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc |  no  |
| [RiskTaxonomy](RiskTaxonomy.md) | A taxonomy of AI system related risks |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |
| [AiTask](AiTask.md) | A task, such as summarization and classification, performed by an AI |  no  |
| [Dataset](Dataset.md) | A body of structured information describing some topic(s) of interest |  no  |
| [Likelihood](Likelihood.md) |  |  no  |
| [IncidentConcludedclass](IncidentConcludedclass.md) |  |  no  |
| [RiskConcept](RiskConcept.md) | An umbrella term for refering to risk, risk source, consequence and impact |  no  |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [RiskControl](RiskControl.md) | A measure that maintains and/or modifies risk (and risk concepts) |  no  |
| [RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |  no  |
| [IncidentNearMissclass](IncidentNearMissclass.md) |  |  no  |
| [AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities |  no  |
| [LargeLanguageModelFamily](LargeLanguageModelFamily.md) | A large language model family is a set of models that are provided by the sam... |  no  |
| [Stakeholder](Stakeholder.md) | An AI system stakeholder for Responsible AI governance (e |  no  |
| [AiEvalResult](AiEvalResult.md) | The result of an evaluation for a specific AI model |  no  |
| [DataPreprocessing](DataPreprocessing.md) | Data transformations, such as PI filtering, performed to ensure high quality ... |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [Modality](Modality.md) | A modality supported by an Ai component |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [IncidentHaltedclass](IncidentHaltedclass.md) |  |  no  |
| [StakeholderGroup](StakeholderGroup.md) | An AI system stakeholder grouping |  no  |
| [AiProvider](AiProvider.md) | A provider under the AI Act is defined by Article 3(3) as a natural or legal ... |  no  |
| [Vocabulary](Vocabulary.md) | A collection of terms, with their definitions and relationships |  no  |
| [Organization](Organization.md) | Any organizational entity such as a corporation, educational institution, con... |  no  |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |
| [RiskIncident](RiskIncident.md) | An event occuring or occured which is a realised or materialised risk |  no  |
| [AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is c... |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [IncidentStatus](IncidentStatus.md) |  |  no  |
| [Term](Term.md) | A term and its definitions |  no  |







## Properties

* Range: [Date](Date.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:dateCreated |
| native | nexus:dateCreated |




## LinkML Source

<details>
```yaml
name: dateCreated
description: The date on which the entity was created.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: schema:dateCreated
alias: dateCreated
domain_of:
- Entity
range: date
required: false

```
</details>
