

# Class: RiskGroup


_A group of AI system related risks that are part of a risk taxonomy._





URI: [nexus:RiskGroup](https://ibm.github.io/risk-atlas-nexus/ontology/RiskGroup)






```mermaid
 classDiagram
    class RiskGroup
    click RiskGroup href "../RiskGroup"
      RiskConcept <|-- RiskGroup
        click RiskConcept href "../RiskConcept"
      Entity <|-- RiskGroup
        click Entity href "../Entity"
      
      RiskGroup : broadMatch
        
          
    
    
    RiskGroup --> "*" Any : broadMatch
    click Any href "../Any"

        
      RiskGroup : closeMatch
        
          
    
    
    RiskGroup --> "*" Any : closeMatch
    click Any href "../Any"

        
      RiskGroup : dateCreated
        
      RiskGroup : dateModified
        
      RiskGroup : description
        
      RiskGroup : exactMatch
        
          
    
    
    RiskGroup --> "*" Any : exactMatch
    click Any href "../Any"

        
      RiskGroup : hasPart
        
          
    
    
    RiskGroup --> "*" Risk : hasPart
    click Risk href "../Risk"

        
      RiskGroup : id
        
      RiskGroup : isDefinedByTaxonomy
        
          
    
    
    RiskGroup --> "0..1" RiskTaxonomy : isDefinedByTaxonomy
    click RiskTaxonomy href "../RiskTaxonomy"

        
      RiskGroup : isDetectedBy
        
          
    
    
    RiskGroup --> "*" RiskControl : isDetectedBy
    click RiskControl href "../RiskControl"

        
      RiskGroup : name
        
      RiskGroup : narrowMatch
        
          
    
    
    RiskGroup --> "*" Any : narrowMatch
    click Any href "../Any"

        
      RiskGroup : relatedMatch
        
          
    
    
    RiskGroup --> "*" Any : relatedMatch
    click Any href "../Any"

        
      RiskGroup : url
        
      
```





## Inheritance
* [Entity](Entity.md)
    * **RiskGroup** [ [RiskConcept](RiskConcept.md)]



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [isDefinedByTaxonomy](isDefinedByTaxonomy.md) | 0..1 <br/> [RiskTaxonomy](RiskTaxonomy.md) | A relationship where a risk or a risk group is defined by a risk taxonomy | direct |
| [closeMatch](closeMatch.md) | * <br/> [Any](Any.md)&nbsp;or&nbsp;<br />[Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskGroup](RiskGroup.md) | The property is used to link two concepts that are sufficiently similar that ... | direct |
| [exactMatch](exactMatch.md) | * <br/> [Any](Any.md)&nbsp;or&nbsp;<br />[Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskGroup](RiskGroup.md) | The property is used to link two concepts, indicating a high degree of confid... | direct |
| [broadMatch](broadMatch.md) | * <br/> [Any](Any.md)&nbsp;or&nbsp;<br />[Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskGroup](RiskGroup.md) | The property is used to state a hierarchical mapping link between two concept... | direct |
| [narrowMatch](narrowMatch.md) | * <br/> [Any](Any.md)&nbsp;or&nbsp;<br />[Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskGroup](RiskGroup.md) | The property is used to state a hierarchical mapping link between two concept... | direct |
| [relatedMatch](relatedMatch.md) | * <br/> [Any](Any.md)&nbsp;or&nbsp;<br />[Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskGroup](RiskGroup.md) | The property skos:relatedMatch is used to state an associative mapping link b... | direct |
| [hasPart](hasPart.md) | * <br/> [Risk](Risk.md) | A relationship where a riskgroup has a risk | direct |
| [isDetectedBy](isDetectedBy.md) | * <br/> [RiskControl](RiskControl.md) | A relationship where a risk, risk source, consequence, or impact is detected ... | [RiskConcept](RiskConcept.md) |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [Container](Container.md) | [riskgroups](riskgroups.md) | range | [RiskGroup](RiskGroup.md) |
| [RiskGroup](RiskGroup.md) | [closeMatch](closeMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |
| [RiskGroup](RiskGroup.md) | [exactMatch](exactMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |
| [RiskGroup](RiskGroup.md) | [broadMatch](broadMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |
| [RiskGroup](RiskGroup.md) | [narrowMatch](narrowMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |
| [RiskGroup](RiskGroup.md) | [relatedMatch](relatedMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |
| [Risk](Risk.md) | [isPartOf](isPartOf.md) | range | [RiskGroup](RiskGroup.md) |
| [Risk](Risk.md) | [closeMatch](closeMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |
| [Risk](Risk.md) | [exactMatch](exactMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |
| [Risk](Risk.md) | [broadMatch](broadMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |
| [Risk](Risk.md) | [narrowMatch](narrowMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |
| [Risk](Risk.md) | [relatedMatch](relatedMatch.md) | any_of[range] | [RiskGroup](RiskGroup.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:RiskGroup |
| native | nexus:RiskGroup |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: RiskGroup
description: A group of AI system related risks that are part of a risk taxonomy.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
mixins:
- RiskConcept
slots:
- isDefinedByTaxonomy
- closeMatch
- exactMatch
- broadMatch
- narrowMatch
- relatedMatch
- hasPart
slot_usage:
  hasPart:
    name: hasPart
    description: A relationship where a riskgroup has a risk
    range: Risk

```
</details>

### Induced

<details>
```yaml
name: RiskGroup
description: A group of AI system related risks that are part of a risk taxonomy.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
mixins:
- RiskConcept
slot_usage:
  hasPart:
    name: hasPart
    description: A relationship where a riskgroup has a risk
    range: Risk
attributes:
  isDefinedByTaxonomy:
    name: isDefinedByTaxonomy
    description: A relationship where a risk or a risk group is defined by a risk
      taxonomy
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:isPartOf
    alias: isDefinedByTaxonomy
    owner: RiskGroup
    domain_of:
    - RiskGroup
    - Risk
    - RiskControl
    - Action
    range: RiskTaxonomy
  closeMatch:
    name: closeMatch
    description: The property is used to link two concepts that are sufficiently similar
      that they can be used interchangeably in some information retrieval applications.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: skos:closeMatch
    alias: closeMatch
    owner: RiskGroup
    domain_of:
    - RiskGroup
    - Risk
    range: Any
    multivalued: true
    inlined: false
    any_of:
    - range: Risk
    - range: RiskGroup
  exactMatch:
    name: exactMatch
    description: The property is used to link two concepts, indicating a high degree
      of confidence that the concepts can be used interchangeably across a wide range
      of information retrieval applications
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: skos:exactMatch
    alias: exactMatch
    owner: RiskGroup
    domain_of:
    - RiskGroup
    - Risk
    range: Any
    multivalued: true
    inlined: false
    any_of:
    - range: Risk
    - range: RiskGroup
  broadMatch:
    name: broadMatch
    description: The property is used to state a hierarchical mapping link between
      two concepts, indicating that the concept linked to, is a broader concept than
      the originating concept.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: skos:broadMatch
    alias: broadMatch
    owner: RiskGroup
    domain_of:
    - RiskGroup
    - Risk
    range: Any
    multivalued: true
    inlined: false
    any_of:
    - range: Risk
    - range: RiskGroup
  narrowMatch:
    name: narrowMatch
    description: The property is used to state a hierarchical mapping link between
      two concepts, indicating that the concept linked to, is a narrower concept than
      the originating concept.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: skos:narrowMatch
    alias: narrowMatch
    owner: RiskGroup
    domain_of:
    - RiskGroup
    - Risk
    range: Any
    multivalued: true
    inlined: false
    any_of:
    - range: Risk
    - range: RiskGroup
  relatedMatch:
    name: relatedMatch
    description: The property skos:relatedMatch is used to state an associative mapping
      link between two concepts.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: skos:relatedMatch
    alias: relatedMatch
    owner: RiskGroup
    domain_of:
    - RiskGroup
    - Risk
    range: Any
    multivalued: true
    inlined: false
    any_of:
    - range: Risk
    - range: RiskGroup
  hasPart:
    name: hasPart
    description: A relationship where a riskgroup has a risk
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:hasPart
    alias: hasPart
    owner: RiskGroup
    domain_of:
    - RiskGroup
    range: Risk
    multivalued: true
  isDetectedBy:
    name: isDetectedBy
    description: A relationship where a risk, risk source, consequence, or impact
      is detected by a risk control.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: isDetectedBy
    owner: RiskGroup
    domain_of:
    - RiskConcept
    inverse: detectsRiskConcept
    range: RiskControl
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
    owner: RiskGroup
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
    owner: RiskGroup
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
    owner: RiskGroup
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
    owner: RiskGroup
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
    owner: RiskGroup
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
    owner: RiskGroup
    domain_of:
    - Entity
    range: date
    required: false

```
</details>