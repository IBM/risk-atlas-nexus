

# Slot: dateCreated


_The date on which the entity was created._





URI: [schema:dateCreated](http://schema.org/dateCreated)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [AiEvalResult](AiEvalResult.md) | The result of an evaluation for a specific AI model |  no  |
| [Entity](Entity.md) | A generic grouping for any identifiable entity |  no  |
| [AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities |  no  |
| [AiLifecyclePhase](AiLifecyclePhase.md) | A Phase of AI lifecycle which indicates evolution of the system from concepti... |  no  |
| [LargeLanguageModelFamily](LargeLanguageModelFamily.md) | A large language model family is a set of models that are provided by the sam... |  no  |
| [IncidentMitigatedclass](IncidentMitigatedclass.md) |  |  no  |
| [IncidentOngoingclass](IncidentOngoingclass.md) |  |  no  |
| [Severity](Severity.md) |  |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [AiTask](AiTask.md) | A task, such as summarization and classification, performed by an AI |  no  |
| [Organization](Organization.md) | Any organizational entity such as a corporation, educational institution, con... |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |
| [BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc |  no  |
| [AiProvider](AiProvider.md) | A provider under the AI Act is defined by Article 3(3) as a natural or legal ... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |
| [IncidentHaltedclass](IncidentHaltedclass.md) |  |  no  |
| [License](License.md) | The general notion of a license which defines terms and grants permissions to... |  no  |
| [IncidentConcludedclass](IncidentConcludedclass.md) |  |  no  |
| [AiModelValidation](AiModelValidation.md) | AI model validation steps that have been performed after the model training t... |  no  |
| [RiskIncident](RiskIncident.md) | An event occuring or occured which is a realised or materialised risk |  no  |
| [IncidentNearMissclass](IncidentNearMissclass.md) |  |  no  |
| [Modality](Modality.md) | A modality supported by an Ai component |  no  |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |
| [Consequence](Consequence.md) |  |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |
| [RiskControl](RiskControl.md) | A measure that maintains and/or modifies risk (and risk concepts) |  no  |
| [Impact](Impact.md) |  |  no  |
| [RiskTaxonomy](RiskTaxonomy.md) | A taxonomy of AI system related risks |  no  |
| [Likelihood](Likelihood.md) |  |  no  |
| [AiOffice](AiOffice.md) | The EU AI Office (https://digital-strategy |  no  |
| [AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is c... |  no  |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [Documentation](Documentation.md) | Documented information about a concept or other topic(s) of interest |  no  |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |  no  |
| [Dataset](Dataset.md) | A body of structured information describing some topic(s) of interest |  no  |
| [DataPreprocessing](DataPreprocessing.md) | Data transformations, such as PI filtering, performed to ensure high quality ... |  no  |
| [IncidentStatus](IncidentStatus.md) |  |  no  |
| [RiskConcept](RiskConcept.md) | An umbrella term for refering to risk, risk source, consequence and impact |  no  |







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