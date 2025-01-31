

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
      Entity <|-- RiskTaxonomy
        click RiskTaxonomy href "../RiskTaxonomy"
      Entity <|-- RiskGroup
        click RiskGroup href "../RiskGroup"
      Entity <|-- Risk
        click Risk href "../Risk"
      Entity <|-- Action
        click Action href "../Action"
      Entity <|-- AiEval
        click AiEval href "../AiEval"
      Entity <|-- AiEvalResult
        click AiEvalResult href "../AiEvalResult"
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
    * [RiskTaxonomy](RiskTaxonomy.md)
    * [RiskGroup](RiskGroup.md)
    * [Risk](Risk.md)
    * [Action](Action.md)
    * [AiEval](AiEval.md)
    * [AiEvalResult](AiEvalResult.md) [ [Fact](Fact.md)]
    * [BaseAi](BaseAi.md)
    * [LargeLanguageModelFamily](LargeLanguageModelFamily.md)
    * [AiTask](AiTask.md)
    * [AiLifecyclePhase](AiLifecyclePhase.md)
    * [Modality](Modality.md)



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


* from schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology




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
from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
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
from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
abstract: true
attributes:
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
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
    from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
    rank: 1000
    slot_uri: schema:name
    alias: name
    owner: Entity
    domain_of:
    - Entity
    range: string
  description:
    name: description
    description: The description of an entity
    from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
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
    from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
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
    from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
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
    from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
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