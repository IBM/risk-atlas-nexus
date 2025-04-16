

# Class: IncidentStatus



URI: [dpv:IncidentStatus](https://w3c.github.io/dpv/2.1/dpv/#IncidentStatus)






```mermaid
 classDiagram
    class IncidentStatus
    click IncidentStatus href "../IncidentStatus"
      Entity <|-- IncidentStatus
        click Entity href "../Entity"
      

      IncidentStatus <|-- IncidentConcludedclass
        click IncidentConcludedclass href "../IncidentConcludedclass"
      IncidentStatus <|-- IncidentHaltedclass
        click IncidentHaltedclass href "../IncidentHaltedclass"
      IncidentStatus <|-- IncidentMitigatedclass
        click IncidentMitigatedclass href "../IncidentMitigatedclass"
      IncidentStatus <|-- IncidentNearMissclass
        click IncidentNearMissclass href "../IncidentNearMissclass"
      IncidentStatus <|-- IncidentOngoingclass
        click IncidentOngoingclass href "../IncidentOngoingclass"
      
      
      IncidentStatus : dateCreated
        
      IncidentStatus : dateModified
        
      IncidentStatus : description
        
      IncidentStatus : id
        
      IncidentStatus : name
        
      IncidentStatus : url
        
      
```





## Inheritance
* [Entity](Entity.md)
    * **IncidentStatus**
        * [IncidentConcludedclass](IncidentConcludedclass.md)
        * [IncidentHaltedclass](IncidentHaltedclass.md)
        * [IncidentMitigatedclass](IncidentMitigatedclass.md)
        * [IncidentNearMissclass](IncidentNearMissclass.md)
        * [IncidentOngoingclass](IncidentOngoingclass.md)



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [RiskIncident](RiskIncident.md) | [hasStatus](hasStatus.md) | range | [IncidentStatus](IncidentStatus.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | dpv:IncidentStatus |
| native | nexus:IncidentStatus |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: IncidentStatus
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
class_uri: dpv:IncidentStatus

```
</details>

### Induced

<details>
```yaml
name: IncidentStatus
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
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
    owner: IncidentStatus
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
    owner: IncidentStatus
    domain_of:
    - Entity
    range: string
  description:
    name: description
    description: The description of an entity
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:description
    alias: description
    owner: IncidentStatus
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
    owner: IncidentStatus
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
    owner: IncidentStatus
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
    owner: IncidentStatus
    domain_of:
    - Entity
    range: date
    required: false
class_uri: dpv:IncidentStatus

```
</details>