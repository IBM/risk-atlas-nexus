

# Class: Term


_A term and its definitions_





URI: [nexus:Term](https://ibm.github.io/risk-atlas-nexus/ontology/Term)






```mermaid
 classDiagram
    class Term
    click Term href "../Term"
      Entity <|-- Term
        click Entity href "../Entity"

      Term : dateCreated

      Term : dateModified

      Term : description

      Term : hasDocumentation





        Term --> "*" Documentation : hasDocumentation
        click Documentation href "../Documentation"



      Term : hasParentDefinition





        Term --> "*" Term : hasParentDefinition
        click Term href "../Term"



      Term : hasRelatedRisk





        Term --> "*" Risk : hasRelatedRisk
        click Risk href "../Risk"



      Term : hasSubDefinition





        Term --> "*" Term : hasSubDefinition
        click Term href "../Term"



      Term : id

      Term : isDefinedByVocabulary





        Term --> "0..1" Vocabulary : isDefinedByVocabulary
        click Vocabulary href "../Vocabulary"



      Term : name

      Term : url


```





## Inheritance
* [Entity](Entity.md)
    * **Term**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [isDefinedByVocabulary](isDefinedByVocabulary.md) | 0..1 <br/> [Vocabulary](Vocabulary.md) | A relationship where a term or a term group is defined by a vocabulary | direct |
| [hasDocumentation](hasDocumentation.md) | * <br/> [Documentation](Documentation.md) | Indicates documentation associated with an entity | direct |
| [hasParentDefinition](hasParentDefinition.md) | * <br/> [Term](Term.md) | Indicates parent terms associated with a term | direct |
| [hasSubDefinition](hasSubDefinition.md) | * <br/> [Term](Term.md) | Indicates child terms associated with a term | direct |
| [hasRelatedRisk](hasRelatedRisk.md) | * <br/> [Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskConcept](RiskConcept.md)&nbsp;or&nbsp;<br />[Term](Term.md) | A relationship where an entity relates to a risk | direct |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [Container](Container.md) | [terms](terms.md) | range | [Term](Term.md) |
| [Term](Term.md) | [hasParentDefinition](hasParentDefinition.md) | range | [Term](Term.md) |
| [Term](Term.md) | [hasSubDefinition](hasSubDefinition.md) | range | [Term](Term.md) |
| [Term](Term.md) | [hasRelatedRisk](hasRelatedRisk.md) | any_of[range] | [Term](Term.md) |
| [LLMQuestionPolicy](LLMQuestionPolicy.md) | [hasRelatedRisk](hasRelatedRisk.md) | any_of[range] | [Term](Term.md) |
| [Action](Action.md) | [hasRelatedRisk](hasRelatedRisk.md) | any_of[range] | [Term](Term.md) |
| [AiEval](AiEval.md) | [hasRelatedRisk](hasRelatedRisk.md) | any_of[range] | [Term](Term.md) |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | [hasRelatedRisk](hasRelatedRisk.md) | any_of[range] | [Term](Term.md) |
| [Question](Question.md) | [hasRelatedRisk](hasRelatedRisk.md) | any_of[range] | [Term](Term.md) |
| [Questionnaire](Questionnaire.md) | [hasRelatedRisk](hasRelatedRisk.md) | any_of[range] | [Term](Term.md) |
| [Adapter](Adapter.md) | [hasRelatedRisk](hasRelatedRisk.md) | any_of[range] | [Term](Term.md) |
| [LLMIntrinsic](LLMIntrinsic.md) | [hasRelatedRisk](hasRelatedRisk.md) | any_of[range] | [Term](Term.md) |
| [LLMIntrinsic](LLMIntrinsic.md) | [hasRelatedTerm](hasRelatedTerm.md) | range | [Term](Term.md) |
| [LLMIntrinsic](LLMIntrinsic.md) | [hasRelatedTerm](hasRelatedTerm.md) | any_of[range] | [Term](Term.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:Term |
| native | nexus:Term |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: Term
description: A term and its definitions
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
slots:
- isDefinedByVocabulary
- hasDocumentation
- hasParentDefinition
- hasSubDefinition
- hasRelatedRisk

```
</details>

### Induced

<details>
```yaml
name: Term
description: A term and its definitions
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
attributes:
  isDefinedByVocabulary:
    name: isDefinedByVocabulary
    description: A relationship where a term or a term group is defined by a vocabulary
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:isPartOf
    alias: isDefinedByVocabulary
    owner: Term
    domain_of:
    - Term
    - Adapter
    - LLMIntrinsic
    range: Vocabulary
  hasDocumentation:
    name: hasDocumentation
    description: Indicates documentation associated with an entity.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:hasDocumentation
    alias: hasDocumentation
    owner: Term
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
  hasParentDefinition:
    name: hasParentDefinition
    description: Indicates parent terms associated with a term
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: nexus:hasParentDefinition
    alias: hasParentDefinition
    owner: Term
    domain_of:
    - Term
    range: Term
    multivalued: true
    inlined: false
  hasSubDefinition:
    name: hasSubDefinition
    description: Indicates child terms associated with a term
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: nexus:hasSubDefinition
    alias: hasSubDefinition
    owner: Term
    domain_of:
    - Term
    range: Term
    multivalued: true
    inlined: false
  hasRelatedRisk:
    name: hasRelatedRisk
    description: A relationship where an entity relates to a risk
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain: Any
    alias: hasRelatedRisk
    owner: Term
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
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: Term
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
    owner: Term
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
    owner: Term
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
    owner: Term
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
    owner: Term
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
    owner: Term
    domain_of:
    - Entity
    range: date
    required: false

```
</details>
