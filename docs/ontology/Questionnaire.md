

# Class: Questionnaire


_A questionnaire groups questions_





URI: [nexus:Questionnaire](https://ibm.github.io/risk-atlas-nexus/ontology/Questionnaire)






```mermaid
 classDiagram
    class Questionnaire
    click Questionnaire href "../Questionnaire"
      AiEval <|-- Questionnaire
        click AiEval href "../AiEval"

      Questionnaire : bestValue

      Questionnaire : dateCreated

      Questionnaire : dateModified

      Questionnaire : description

      Questionnaire : hasBenchmarkMetadata





        Questionnaire --> "*" BenchmarkMetadataCard : hasBenchmarkMetadata
        click BenchmarkMetadataCard href "../BenchmarkMetadataCard"



      Questionnaire : hasDataset





        Questionnaire --> "*" Dataset : hasDataset
        click Dataset href "../Dataset"



      Questionnaire : hasDocumentation





        Questionnaire --> "*" Documentation : hasDocumentation
        click Documentation href "../Documentation"



      Questionnaire : hasImplementation

      Questionnaire : hasLicense





        Questionnaire --> "0..1" License : hasLicense
        click License href "../License"



      Questionnaire : hasRelatedRisk





        Questionnaire --> "*" Risk : hasRelatedRisk
        click Risk href "../Risk"



      Questionnaire : hasTasks

      Questionnaire : hasUnitxtCard

      Questionnaire : id

      Questionnaire : name

      Questionnaire : url


```





## Inheritance
* [Entity](Entity.md)
    * [AiEval](AiEval.md)
        * **Questionnaire**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [hasDocumentation](hasDocumentation.md) | * <br/> [Documentation](Documentation.md) | Indicates documentation associated with an entity | [AiEval](AiEval.md) |
| [hasDataset](hasDataset.md) | * <br/> [Dataset](Dataset.md) | A relationship to datasets that are used | [AiEval](AiEval.md) |
| [hasTasks](hasTasks.md) | * <br/> [String](String.md) | The tasks or evaluations the benchmark is intended to assess | [AiEval](AiEval.md) |
| [hasImplementation](hasImplementation.md) | * <br/> [Uri](Uri.md) | A relationship to a implementation defining the risk evaluation | [AiEval](AiEval.md) |
| [hasUnitxtCard](hasUnitxtCard.md) | * <br/> [Uri](Uri.md) | A relationship to a Unitxt card defining the risk evaluation | [AiEval](AiEval.md) |
| [hasLicense](hasLicense.md) | 0..1 <br/> [License](License.md) | Indicates licenses associated with a resource | [AiEval](AiEval.md) |
| [hasRelatedRisk](hasRelatedRisk.md) | * <br/> [Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskConcept](RiskConcept.md)&nbsp;or&nbsp;<br />[Term](Term.md) | A relationship where an entity relates to a risk | [AiEval](AiEval.md) |
| [bestValue](bestValue.md) | 0..1 <br/> [String](String.md) | Annotation of the best possible result of the evaluation | [AiEval](AiEval.md) |
| [hasBenchmarkMetadata](hasBenchmarkMetadata.md) | * <br/> [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | A relationship to a Benchmark Metadata Card which contains metadata about the... | [AiEval](AiEval.md) |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |









## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:Questionnaire |
| native | nexus:Questionnaire |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: Questionnaire
description: A questionnaire groups questions
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: AiEval
slot_usage:
  composed_of:
    name: composed_of
    range: Question

```
</details>

### Induced

<details>
```yaml
name: Questionnaire
description: A questionnaire groups questions
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: AiEval
slot_usage:
  composed_of:
    name: composed_of
    range: Question
attributes:
  hasDocumentation:
    name: hasDocumentation
    description: Indicates documentation associated with an entity.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:hasDocumentation
    alias: hasDocumentation
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
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
    owner: Questionnaire
    domain_of:
    - Entity
    range: date
    required: false

```
</details>
