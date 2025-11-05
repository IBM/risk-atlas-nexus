

# Class: Entity


_A generic grouping for any identifiable entity._




* __NOTE__: this is an abstract class and should not be instantiated directly


URI: [schema:Thing](http://schema.org/Thing)






```mermaid
 classDiagram
    class Entity
    click Entity href "../Entity"
      Entity <|-- Organization
        click Organization href "../Organization"
      Entity <|-- License
        click License href "../License"
      Entity <|-- Dataset
        click Dataset href "../Dataset"
      Entity <|-- Documentation
        click Documentation href "../Documentation"
      Entity <|-- Vocabulary
        click Vocabulary href "../Vocabulary"
      Entity <|-- Term
        click Term href "../Term"
      Entity <|-- Principle
        click Principle href "../Principle"
      Entity <|-- Policy
        click Policy href "../Policy"
      Entity <|-- Rule
        click Rule href "../Rule"
      Entity <|-- RiskTaxonomy
        click RiskTaxonomy href "../RiskTaxonomy"
      Entity <|-- RiskGroup
        click RiskGroup href "../RiskGroup"
      Entity <|-- Risk
        click Risk href "../Risk"
      Entity <|-- RiskConcept
        click RiskConcept href "../RiskConcept"
      Entity <|-- RiskControl
        click RiskControl href "../RiskControl"
      Entity <|-- RiskIncident
        click RiskIncident href "../RiskIncident"
      Entity <|-- Impact
        click Impact href "../Impact"
      Entity <|-- IncidentStatus
        click IncidentStatus href "../IncidentStatus"
      Entity <|-- Severity
        click Severity href "../Severity"
      Entity <|-- Likelihood
        click Likelihood href "../Likelihood"
      Entity <|-- Consequence
        click Consequence href "../Consequence"
      Entity <|-- BaseAi
        click BaseAi href "../BaseAi"
      Entity <|-- LargeLanguageModelFamily
        click LargeLanguageModelFamily href "../LargeLanguageModelFamily"
      Entity <|-- AiTask
        click AiTask href "../AiTask"
      Entity <|-- AiLifecyclePhase
        click AiLifecyclePhase href "../AiLifecyclePhase"
      Entity <|-- Modality
        click Modality href "../Modality"
      Entity <|-- Input
        click Input href "../Input"
      Entity <|-- AiEval
        click AiEval href "../AiEval"
      Entity <|-- AiEvalResult
        click AiEvalResult href "../AiEvalResult"
      Entity <|-- BenchmarkMetadataCard
        click BenchmarkMetadataCard href "../BenchmarkMetadataCard"
      Entity <|-- Adapter
        click Adapter href "../Adapter"
      Entity <|-- LLMIntrinsic
        click LLMIntrinsic href "../LLMIntrinsic"
      Entity <|-- StakeholderGroup
        click StakeholderGroup href "../StakeholderGroup"
      Entity <|-- Stakeholder
        click Stakeholder href "../Stakeholder"

      Entity : dateCreated

      Entity : dateModified

      Entity : description

      Entity : id

      Entity : name

      Entity : url


```





## Inheritance
* **Entity**
    * [Organization](Organization.md)
    * [License](License.md)
    * [Dataset](Dataset.md)
    * [Documentation](Documentation.md)
    * [Vocabulary](Vocabulary.md)
    * [Term](Term.md)
    * [Principle](Principle.md)
    * [Policy](Policy.md)
    * [Rule](Rule.md)
    * [RiskTaxonomy](RiskTaxonomy.md)
    * [RiskGroup](RiskGroup.md) [ [RiskConcept](RiskConcept.md)]
    * [Risk](Risk.md) [ [RiskConcept](RiskConcept.md)]
    * [RiskConcept](RiskConcept.md)
    * [RiskControl](RiskControl.md) [ [RiskConcept](RiskConcept.md)]
    * [RiskIncident](RiskIncident.md) [ [RiskConcept](RiskConcept.md)]
    * [Impact](Impact.md) [ [RiskConcept](RiskConcept.md)]
    * [IncidentStatus](IncidentStatus.md)
    * [Severity](Severity.md)
    * [Likelihood](Likelihood.md)
    * [Consequence](Consequence.md)
    * [BaseAi](BaseAi.md)
    * [LargeLanguageModelFamily](LargeLanguageModelFamily.md)
    * [AiTask](AiTask.md)
    * [AiLifecyclePhase](AiLifecyclePhase.md)
    * [Modality](Modality.md)
    * [Input](Input.md)
    * [AiEval](AiEval.md)
    * [AiEvalResult](AiEvalResult.md) [ [Fact](Fact.md)]
    * [BenchmarkMetadataCard](BenchmarkMetadataCard.md)
    * [Adapter](Adapter.md) [ [LargeLanguageModel](LargeLanguageModel.md)]
    * [LLMIntrinsic](LLMIntrinsic.md)
    * [StakeholderGroup](StakeholderGroup.md)
    * [Stakeholder](Stakeholder.md)



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | direct |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | direct |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | direct |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | direct |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | direct |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | direct |









## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:Thing |
| native | nexus:Entity |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: Entity
description: A generic grouping for any identifiable entity.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
abstract: true
slots:
- id
- name
- description
- url
- dateCreated
- dateModified
class_uri: schema:Thing

```
</details>

### Induced

<details>
```yaml
name: Entity
description: A generic grouping for any identifiable entity.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
abstract: true
attributes:
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: Entity
    domain_of:
    - Entity
    range: string
    required: true
  name:
    name: name
    description: A text name of this instance.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:name
    alias: name
    owner: Entity
    domain_of:
    - Entity
    - BenchmarkMetadataCard
    range: string
  description:
    name: description
    description: The description of an entity
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:description
    alias: description
    owner: Entity
    domain_of:
    - Entity
    range: string
  url:
    name: url
    description: An optional URL associated with this instance.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:url
    alias: url
    owner: Entity
    domain_of:
    - Entity
    range: uri
  dateCreated:
    name: dateCreated
    description: The date on which the entity was created.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:dateCreated
    alias: dateCreated
    owner: Entity
    domain_of:
    - Entity
    range: date
    required: false
  dateModified:
    name: dateModified
    description: The date on which the entity was most recently modified.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:dateModified
    alias: dateModified
    owner: Entity
    domain_of:
    - Entity
    range: date
    required: false
class_uri: schema:Thing

```
</details>
