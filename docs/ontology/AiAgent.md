

# Class: AiAgent


_An artificial intelligence (AI) agent refers to a system or program that is capable of autonomously performing tasks on behalf of a user or another system by designing its workflow and utilizing available tools._





URI: [nexus:AiAgent](https://ibm.github.io/risk-atlas-nexus/ontology/AiAgent)






```mermaid
 classDiagram
    class AiAgent
    click AiAgent href "../AiAgent"
      AiSystem <|-- AiAgent
        click AiSystem href "../AiSystem"

      AiAgent : dateCreated

      AiAgent : dateModified

      AiAgent : description

      AiAgent : hasDocumentation





        AiAgent --> "*" Documentation : hasDocumentation
        click Documentation href "../Documentation"



      AiAgent : hasEuAiSystemType





        AiAgent --> "0..1" AiSystemType : hasEuAiSystemType
        click AiSystemType href "../AiSystemType"



      AiAgent : hasEuRiskCategory





        AiAgent --> "0..1" EuAiRiskCategory : hasEuRiskCategory
        click EuAiRiskCategory href "../EuAiRiskCategory"



      AiAgent : hasLicense





        AiAgent --> "0..1" License : hasLicense
        click License href "../License"



      AiAgent : hasModelCard

      AiAgent : id

      AiAgent : isProvidedBy





        AiAgent --> "0..1" AiProvider : isProvidedBy
        click AiProvider href "../AiProvider"



      AiAgent : name

      AiAgent : performsTask





        AiAgent --> "*" AiTask : performsTask
        click AiTask href "../AiTask"



      AiAgent : producer





        AiAgent --> "0..1" Organization : producer
        click Organization href "../Organization"



      AiAgent : url


```





## Inheritance
* [Entity](Entity.md)
    * [BaseAi](BaseAi.md)
        * [AiSystem](AiSystem.md)
            * **AiAgent**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [hasEuAiSystemType](hasEuAiSystemType.md) | 0..1 <br/> [AiSystemType](AiSystemType.md) | The type of system as defined by the EU AI Act | [AiSystem](AiSystem.md) |
| [hasEuRiskCategory](hasEuRiskCategory.md) | 0..1 <br/> [EuAiRiskCategory](EuAiRiskCategory.md) | The risk category of an AI system as defined by the EU AI Act | [AiSystem](AiSystem.md) |
| [producer](producer.md) | 0..1 <br/> [Organization](Organization.md) | A relationship to the Organization instance which produces this instance | [BaseAi](BaseAi.md) |
| [hasModelCard](hasModelCard.md) | * <br/> [String](String.md) | A relationship to model card references | [BaseAi](BaseAi.md) |
| [hasDocumentation](hasDocumentation.md) | * <br/> [Documentation](Documentation.md) | Indicates documentation associated with an entity | [BaseAi](BaseAi.md) |
| [hasLicense](hasLicense.md) | 0..1 <br/> [License](License.md) | Indicates licenses associated with a resource | [BaseAi](BaseAi.md) |
| [performsTask](performsTask.md) | * <br/> [AiTask](AiTask.md) | relationship indicating the AI tasks an AI model can perform | [BaseAi](BaseAi.md) |
| [isProvidedBy](isProvidedBy.md) | 0..1 <br/> [AiProvider](AiProvider.md) | A relationship indicating the AI agent has been provided by an AI systems pro... | [BaseAi](BaseAi.md) |
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
| self | nexus:AiAgent |
| native | nexus:AiAgent |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: AiAgent
description: An artificial intelligence (AI) agent refers to a system or program that
  is capable of autonomously performing tasks on behalf of a user or another system
  by designing its workflow and utilizing available tools.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: AiSystem
slot_usage:
  isProvidedBy:
    name: isProvidedBy
    description: A relationship indicating the AI agent has been provided by an AI
      systems provider.

```
</details>

### Induced

<details>
```yaml
name: AiAgent
description: An artificial intelligence (AI) agent refers to a system or program that
  is capable of autonomously performing tasks on behalf of a user or another system
  by designing its workflow and utilizing available tools.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: AiSystem
slot_usage:
  isProvidedBy:
    name: isProvidedBy
    description: A relationship indicating the AI agent has been provided by an AI
      systems provider.
attributes:
  hasEuAiSystemType:
    name: hasEuAiSystemType
    description: The type of system as defined by the EU AI Act.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: hasEuAiSystemType
    owner: AiAgent
    domain_of:
    - AiSystem
    range: AiSystemType
  hasEuRiskCategory:
    name: hasEuRiskCategory
    description: The risk category of an AI system as defined by the EU AI Act.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: hasEuRiskCategory
    owner: AiAgent
    domain_of:
    - AiSystem
    range: EuAiRiskCategory
  producer:
    name: producer
    description: A relationship to the Organization instance which produces this instance.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: producer
    owner: AiAgent
    domain_of:
    - BaseAi
    range: Organization
  hasModelCard:
    name: hasModelCard
    description: A relationship to model card references.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: hasModelCard
    owner: AiAgent
    domain_of:
    - BaseAi
    range: string
    multivalued: true
    inlined: true
    inlined_as_list: true
  hasDocumentation:
    name: hasDocumentation
    description: Indicates documentation associated with an entity.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:hasDocumentation
    alias: hasDocumentation
    owner: AiAgent
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
  hasLicense:
    name: hasLicense
    description: Indicates licenses associated with a resource
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:hasLicense
    alias: hasLicense
    owner: AiAgent
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
  performsTask:
    name: performsTask
    description: relationship indicating the AI tasks an AI model can perform.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: performsTask
    owner: AiAgent
    domain_of:
    - BaseAi
    range: AiTask
    multivalued: true
    inlined: false
  isProvidedBy:
    name: isProvidedBy
    description: A relationship indicating the AI agent has been provided by an AI
      systems provider.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:isProvidedBy
    alias: isProvidedBy
    owner: AiAgent
    domain_of:
    - BaseAi
    range: AiProvider
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: AiAgent
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
    owner: AiAgent
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
    owner: AiAgent
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
    owner: AiAgent
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
    owner: AiAgent
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
    owner: AiAgent
    domain_of:
    - Entity
    range: date
    required: false

```
</details>
