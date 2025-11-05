

# Class: AiEval


_An AI Evaluation, e.g. a metric, benchmark, unitxt card evaluation, a question or a combination of such entities._





URI: [dqv:Metric](https://www.w3.org/TR/vocab-dqv/Metric)






```mermaid
 classDiagram
    class AiEval
    click AiEval href "../AiEval"
      Entity <|-- AiEval
        click Entity href "../Entity"


      AiEval <|-- Question
        click Question href "../Question"
      AiEval <|-- Questionnaire
        click Questionnaire href "../Questionnaire"


      AiEval : bestValue

      AiEval : dateCreated

      AiEval : dateModified

      AiEval : description

      AiEval : hasBenchmarkMetadata





        AiEval --> "*" BenchmarkMetadataCard : hasBenchmarkMetadata
        click BenchmarkMetadataCard href "../BenchmarkMetadataCard"



      AiEval : hasDataset





        AiEval --> "*" Dataset : hasDataset
        click Dataset href "../Dataset"



      AiEval : hasDocumentation





        AiEval --> "*" Documentation : hasDocumentation
        click Documentation href "../Documentation"



      AiEval : hasImplementation

      AiEval : hasLicense





        AiEval --> "0..1" License : hasLicense
        click License href "../License"



      AiEval : hasRelatedRisk





        AiEval --> "*" Risk : hasRelatedRisk
        click Risk href "../Risk"



      AiEval : hasTasks

      AiEval : hasUnitxtCard

      AiEval : id

      AiEval : name

      AiEval : url


```





## Inheritance
* [Entity](Entity.md)
    * **AiEval**
        * [Question](Question.md)
        * [Questionnaire](Questionnaire.md)



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [hasDocumentation](hasDocumentation.md) | * <br/> [Documentation](Documentation.md) | Indicates documentation associated with an entity | direct |
| [hasDataset](hasDataset.md) | * <br/> [Dataset](Dataset.md) | A relationship to datasets that are used | direct |
| [hasTasks](hasTasks.md) | * <br/> [String](String.md) | The tasks or evaluations the benchmark is intended to assess | direct |
| [hasImplementation](hasImplementation.md) | * <br/> [Uri](Uri.md) | A relationship to a implementation defining the risk evaluation | direct |
| [hasUnitxtCard](hasUnitxtCard.md) | * <br/> [Uri](Uri.md) | A relationship to a Unitxt card defining the risk evaluation | direct |
| [hasLicense](hasLicense.md) | 0..1 <br/> [License](License.md) | Indicates licenses associated with a resource | direct |
| [hasRelatedRisk](hasRelatedRisk.md) | * <br/> [Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskConcept](RiskConcept.md)&nbsp;or&nbsp;<br />[Term](Term.md) | A relationship where an entity relates to a risk | direct |
| [bestValue](bestValue.md) | 0..1 <br/> [String](String.md) | Annotation of the best possible result of the evaluation | direct |
| [hasBenchmarkMetadata](hasBenchmarkMetadata.md) | * <br/> [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | A relationship to a Benchmark Metadata Card which contains metadata about the... | direct |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [Container](Container.md) | [evaluations](evaluations.md) | range | [AiEval](AiEval.md) |
| [AiEval](AiEval.md) | [hasBenchmarkMetadata](hasBenchmarkMetadata.md) | domain | [AiEval](AiEval.md) |
| [AiEvalResult](AiEvalResult.md) | [isResultOf](isResultOf.md) | range | [AiEval](AiEval.md) |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | [describesAiEval](describesAiEval.md) | range | [AiEval](AiEval.md) |
| [Question](Question.md) | [hasBenchmarkMetadata](hasBenchmarkMetadata.md) | domain | [AiEval](AiEval.md) |
| [Questionnaire](Questionnaire.md) | [hasBenchmarkMetadata](hasBenchmarkMetadata.md) | domain | [AiEval](AiEval.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | dqv:Metric |
| native | nexus:AiEval |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: AiEval
description: An AI Evaluation, e.g. a metric, benchmark, unitxt card evaluation, a
  question or a combination of such entities.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
slots:
- hasDocumentation
- hasDataset
- hasTasks
- hasImplementation
- hasUnitxtCard
- hasLicense
- hasRelatedRisk
- bestValue
- hasBenchmarkMetadata
slot_usage:
  isComposedOf:
    name: isComposedOf
    description: A relationship indicating that an AI evaluation maybe composed of
      other AI evaluations (e.g. it's an overall average of other scores).
    range: AiEval
class_uri: dqv:Metric

```
</details>

### Induced

<details>
```yaml
name: AiEval
description: An AI Evaluation, e.g. a metric, benchmark, unitxt card evaluation, a
  question or a combination of such entities.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
slot_usage:
  isComposedOf:
    name: isComposedOf
    description: A relationship indicating that an AI evaluation maybe composed of
      other AI evaluations (e.g. it's an overall average of other scores).
    range: AiEval
attributes:
  hasDocumentation:
    name: hasDocumentation
    description: Indicates documentation associated with an entity.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:hasDocumentation
    alias: hasDocumentation
    owner: AiEval
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
  hasDataset:
    name: hasDataset
    description: A relationship to datasets that are used.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: hasDataset
    owner: AiEval
    domain_of:
    - AiEval
    range: Dataset
    multivalued: true
    inlined: false
  hasTasks:
    name: hasTasks
    description: The tasks or evaluations the benchmark is intended to assess.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: hasTasks
    owner: AiEval
    domain_of:
    - AiEval
    - BenchmarkMetadataCard
    range: string
    multivalued: true
    inlined: false
  hasImplementation:
    name: hasImplementation
    description: A relationship to a implementation defining the risk evaluation
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:url
    alias: hasImplementation
    owner: AiEval
    domain_of:
    - AiEval
    range: uri
    multivalued: true
    inlined: false
  hasUnitxtCard:
    name: hasUnitxtCard
    description: A relationship to a Unitxt card defining the risk evaluation
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:url
    alias: hasUnitxtCard
    owner: AiEval
    domain_of:
    - AiEval
    range: uri
    multivalued: true
    inlined: false
  hasLicense:
    name: hasLicense
    description: Indicates licenses associated with a resource
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:hasLicense
    alias: hasLicense
    owner: AiEval
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
  hasRelatedRisk:
    name: hasRelatedRisk
    description: A relationship where an entity relates to a risk
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain: Any
    alias: hasRelatedRisk
    owner: AiEval
    domain_of:
    - Term
    - LLMQuestionPolicy
    - Action
    - AiEval
    - BenchmarkMetadataCard
    - Adapter
    - LLMIntrinsic
    range: Risk
    multivalued: true
    inlined: false
    any_of:
    - range: RiskConcept
    - range: Term
  bestValue:
    name: bestValue
    description: Annotation of the best possible result of the evaluation
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: bestValue
    owner: AiEval
    domain_of:
    - AiEval
    range: string
  hasBenchmarkMetadata:
    name: hasBenchmarkMetadata
    description: A relationship to a Benchmark Metadata Card which contains metadata
      about the benchmark.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain: AiEval
    alias: hasBenchmarkMetadata
    owner: AiEval
    domain_of:
    - AiEval
    inverse: describesAiEval
    range: BenchmarkMetadataCard
    multivalued: true
    inlined: false
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: AiEval
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
    owner: AiEval
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
    owner: AiEval
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
    owner: AiEval
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
    owner: AiEval
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
    owner: AiEval
    domain_of:
    - Entity
    range: date
    required: false
class_uri: dqv:Metric

```
</details>
