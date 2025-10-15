

# Class: Stakeholder


_An AI system stakeholder for Responsible AI governance (e.g., AI governors, users, consumers)._





URI: [nexus:Stakeholder](https://ibm.github.io/risk-atlas-nexus/ontology/Stakeholder)






```mermaid
 classDiagram
    class Stakeholder
    click Stakeholder href "../Stakeholder"
      Entity <|-- Stakeholder
        click Entity href "../Entity"

      Stakeholder : dateCreated

      Stakeholder : dateModified

      Stakeholder : description

      Stakeholder : id

      Stakeholder : isDefinedByTaxonomy





        Stakeholder --> "0..1" RiskTaxonomy : isDefinedByTaxonomy
        click RiskTaxonomy href "../RiskTaxonomy"



      Stakeholder : isPartOf





        Stakeholder --> "0..1" StakeholderGroup : isPartOf
        click StakeholderGroup href "../StakeholderGroup"



      Stakeholder : name

      Stakeholder : url


```





## Inheritance
* [Entity](Entity.md)
    * **Stakeholder**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [isDefinedByTaxonomy](isDefinedByTaxonomy.md) | 0..1 <br/> [RiskTaxonomy](RiskTaxonomy.md) | A relationship where a risk or a risk group is defined by a risk taxonomy | direct |
| [isPartOf](isPartOf.md) | 0..1 <br/> [StakeholderGroup](StakeholderGroup.md) | A relationship where a stakeholder is part of a stakeholder group | direct |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [Container](Container.md) | [stakeholders](stakeholders.md) | range | [Stakeholder](Stakeholder.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:Stakeholder |
| native | nexus:Stakeholder |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: Stakeholder
description: An AI system stakeholder for Responsible AI governance (e.g., AI governors,
  users, consumers).
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
slots:
- isDefinedByTaxonomy
- isPartOf
slot_usage:
  isPartOf:
    name: isPartOf
    description: A relationship where a stakeholder is part of a stakeholder group
    range: StakeholderGroup

```
</details>

### Induced

<details>
```yaml
name: Stakeholder
description: An AI system stakeholder for Responsible AI governance (e.g., AI governors,
  users, consumers).
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
slot_usage:
  isPartOf:
    name: isPartOf
    description: A relationship where a stakeholder is part of a stakeholder group
    range: StakeholderGroup
attributes:
  isDefinedByTaxonomy:
    name: isDefinedByTaxonomy
    description: A relationship where a risk or a risk group is defined by a risk
      taxonomy
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:isPartOf
    alias: isDefinedByTaxonomy
    owner: Stakeholder
    domain_of:
    - Policy
    - RiskGroup
    - Risk
    - RiskControl
    - Action
    - RiskIncident
    - StakeholderGroup
    - Stakeholder
    range: RiskTaxonomy
  isPartOf:
    name: isPartOf
    description: A relationship where a stakeholder is part of a stakeholder group
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:isPartOf
    alias: isPartOf
    owner: Stakeholder
    domain_of:
    - Risk
    - LargeLanguageModel
    - Stakeholder
    range: StakeholderGroup
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: Stakeholder
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
    owner: Stakeholder
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
    owner: Stakeholder
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
    owner: Stakeholder
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
    owner: Stakeholder
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
    owner: Stakeholder
    domain_of:
    - Entity
    range: date
    required: false

```
</details>
