

# Class: LargeLanguageModel


_A large language model (LLM) is an AI model which supports a range of language-related tasks such as generation, summarization, classification, among others. A LLM is implemented as an artificial neural networks using a transformer architecture._





URI: [nexus:LargeLanguageModel](https://ibm.github.io/risk-atlas-nexus/ontology/LargeLanguageModel)






```mermaid
 classDiagram
    class LargeLanguageModel
    click LargeLanguageModel href "../LargeLanguageModel"
      AiModel <|-- LargeLanguageModel
        click AiModel href "../AiModel"


      LargeLanguageModel <|-- Adapter
        click Adapter href "../Adapter"


      LargeLanguageModel : architecture

      LargeLanguageModel : carbon_emitted

      LargeLanguageModel : contextWindowSize

      LargeLanguageModel : dateCreated

      LargeLanguageModel : dateModified

      LargeLanguageModel : description

      LargeLanguageModel : fine_tuning

      LargeLanguageModel : gpu_hours

      LargeLanguageModel : hasDocumentation





        LargeLanguageModel --> "*" Documentation : hasDocumentation
        click Documentation href "../Documentation"



      LargeLanguageModel : hasEvaluation





        LargeLanguageModel --> "*" AiEvalResult : hasEvaluation
        click AiEvalResult href "../AiEvalResult"



      LargeLanguageModel : hasInputModality





        LargeLanguageModel --> "*" Modality : hasInputModality
        click Modality href "../Modality"



      LargeLanguageModel : hasLicense





        LargeLanguageModel --> "0..1" License : hasLicense
        click License href "../License"



      LargeLanguageModel : hasModelCard

      LargeLanguageModel : hasOutputModality





        LargeLanguageModel --> "*" Modality : hasOutputModality
        click Modality href "../Modality"



      LargeLanguageModel : hasRiskControl





        LargeLanguageModel --> "*" RiskControl : hasRiskControl
        click RiskControl href "../RiskControl"



      LargeLanguageModel : hasTrainingData





        LargeLanguageModel --> "*" Dataset : hasTrainingData
        click Dataset href "../Dataset"



      LargeLanguageModel : id

      LargeLanguageModel : isPartOf





        LargeLanguageModel --> "0..1" LargeLanguageModelFamily : isPartOf
        click LargeLanguageModelFamily href "../LargeLanguageModelFamily"



      LargeLanguageModel : isProvidedBy





        LargeLanguageModel --> "0..1" AiProvider : isProvidedBy
        click AiProvider href "../AiProvider"



      LargeLanguageModel : name

      LargeLanguageModel : numParameters

      LargeLanguageModel : numTrainingTokens

      LargeLanguageModel : performsTask





        LargeLanguageModel --> "*" AiTask : performsTask
        click AiTask href "../AiTask"



      LargeLanguageModel : power_consumption_w

      LargeLanguageModel : producer





        LargeLanguageModel --> "0..1" Organization : producer
        click Organization href "../Organization"



      LargeLanguageModel : supported_languages

      LargeLanguageModel : url


```





## Inheritance
* [Entity](Entity.md)
    * [BaseAi](BaseAi.md)
        * [AiModel](AiModel.md)
            * **LargeLanguageModel**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [numParameters](numParameters.md) | 0..1 <br/> [Integer](Integer.md) | A property indicating the number of parameters in a LLM | direct |
| [numTrainingTokens](numTrainingTokens.md) | 0..1 <br/> [Integer](Integer.md) | The number of tokens a AI model was trained on | direct |
| [contextWindowSize](contextWindowSize.md) | 0..1 <br/> [Integer](Integer.md) | The total length, in bytes, of an AI model's context window | direct |
| [hasInputModality](hasInputModality.md) | * <br/> [Modality](Modality.md) | A relationship indicating the input modalities supported by an AI component | direct |
| [hasOutputModality](hasOutputModality.md) | * <br/> [Modality](Modality.md) | A relationship indicating the output modalities supported by an AI component | direct |
| [hasTrainingData](hasTrainingData.md) | * <br/> [Dataset](Dataset.md) | A relationship indicating the datasets an AI model was trained on | direct |
| [fine_tuning](fine_tuning.md) | 0..1 <br/> [String](String.md) | A description of the fine-tuning mechanism(s) applied to a model | direct |
| [supported_languages](supported_languages.md) | * <br/> [String](String.md) | A list of languages, expressed as ISO two letter codes | direct |
| [isPartOf](isPartOf.md) | 0..1 <br/> [LargeLanguageModelFamily](LargeLanguageModelFamily.md) | Annotation that a Large Language model is part of a family of models | direct |
| [hasEvaluation](hasEvaluation.md) | * <br/> [AiEvalResult](AiEvalResult.md) | A relationship indicating that an entity has an AI evaluation result | [AiModel](AiModel.md) |
| [architecture](architecture.md) | 0..1 <br/> [String](String.md) | A description of the architecture of an AI such as 'Decoder-only' | [AiModel](AiModel.md) |
| [gpu_hours](gpu_hours.md) | 0..1 <br/> [Integer](Integer.md) | GPU consumption in terms of hours | [AiModel](AiModel.md) |
| [power_consumption_w](power_consumption_w.md) | 0..1 <br/> [Integer](Integer.md) | power consumption in Watts | [AiModel](AiModel.md) |
| [carbon_emitted](carbon_emitted.md) | 0..1 <br/> [Float](Float.md) | The number of tons of carbon dioxide equivalent that are emitted during train... | [AiModel](AiModel.md) |
| [hasRiskControl](hasRiskControl.md) | * <br/> [RiskControl](RiskControl.md) | Indicates the control measures associated with a system or component to modif... | [AiModel](AiModel.md) |
| [producer](producer.md) | 0..1 <br/> [Organization](Organization.md) | A relationship to the Organization instance which produces this instance | [BaseAi](BaseAi.md) |
| [hasModelCard](hasModelCard.md) | * <br/> [String](String.md) | A relationship to model card references | [BaseAi](BaseAi.md) |
| [hasDocumentation](hasDocumentation.md) | * <br/> [Documentation](Documentation.md) | Indicates documentation associated with an entity | [BaseAi](BaseAi.md) |
| [hasLicense](hasLicense.md) | 0..1 <br/> [License](License.md) | Indicates licenses associated with a resource | [BaseAi](BaseAi.md) |
| [performsTask](performsTask.md) | * <br/> [AiTask](AiTask.md) | relationship indicating the AI tasks an AI model can perform | [BaseAi](BaseAi.md) |
| [isProvidedBy](isProvidedBy.md) | 0..1 <br/> [AiProvider](AiProvider.md) | A relationship indicating the AI model has been provided by an AI model provi... | [BaseAi](BaseAi.md) |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [Container](Container.md) | [aimodels](aimodels.md) | range | [LargeLanguageModel](LargeLanguageModel.md) |
| [Adapter](Adapter.md) | [adaptsModel](adaptsModel.md) | range | [LargeLanguageModel](LargeLanguageModel.md) |




## Aliases


* LLM



## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:LargeLanguageModel |
| native | nexus:LargeLanguageModel |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: LargeLanguageModel
description: A large language model (LLM) is an AI model which supports a range of
  language-related tasks such as generation, summarization, classification, among
  others. A LLM is implemented as an artificial neural networks using a transformer
  architecture.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
aliases:
- LLM
is_a: AiModel
slots:
- numParameters
- numTrainingTokens
- contextWindowSize
- hasInputModality
- hasOutputModality
- hasTrainingData
- fine_tuning
- supported_languages
- isPartOf
slot_usage:
  isPartOf:
    name: isPartOf
    description: Annotation that a Large Language model is part of a family of models
    range: LargeLanguageModelFamily

```
</details>

### Induced

<details>
```yaml
name: LargeLanguageModel
description: A large language model (LLM) is an AI model which supports a range of
  language-related tasks such as generation, summarization, classification, among
  others. A LLM is implemented as an artificial neural networks using a transformer
  architecture.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
aliases:
- LLM
is_a: AiModel
slot_usage:
  isPartOf:
    name: isPartOf
    description: Annotation that a Large Language model is part of a family of models
    range: LargeLanguageModelFamily
attributes:
  numParameters:
    name: numParameters
    description: A property indicating the number of parameters in a LLM.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: numParameters
    owner: LargeLanguageModel
    domain_of:
    - LargeLanguageModel
    range: integer
    minimum_value: 0
  numTrainingTokens:
    name: numTrainingTokens
    description: The number of tokens a AI model was trained on.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: numTrainingTokens
    owner: LargeLanguageModel
    domain_of:
    - LargeLanguageModel
    range: integer
    minimum_value: 0
  contextWindowSize:
    name: contextWindowSize
    description: The total length, in bytes, of an AI model's context window.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: contextWindowSize
    owner: LargeLanguageModel
    domain_of:
    - LargeLanguageModel
    range: integer
    minimum_value: 0
  hasInputModality:
    name: hasInputModality
    description: A relationship indicating the input modalities supported by an AI
      component. Examples include text, image, video.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: hasInputModality
    owner: LargeLanguageModel
    domain_of:
    - LargeLanguageModel
    range: Modality
    multivalued: true
    inlined: false
  hasOutputModality:
    name: hasOutputModality
    description: A relationship indicating the output modalities supported by an AI
      component. Examples include text, image, video.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: hasOutputModality
    owner: LargeLanguageModel
    domain_of:
    - LargeLanguageModel
    range: Modality
    multivalued: true
    inlined: false
  hasTrainingData:
    name: hasTrainingData
    description: A relationship indicating the datasets an AI model was trained on.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:hasTrainingData
    alias: hasTrainingData
    owner: LargeLanguageModel
    domain_of:
    - LargeLanguageModel
    range: Dataset
    multivalued: true
    inlined: false
  fine_tuning:
    name: fine_tuning
    description: A description of the fine-tuning mechanism(s) applied to a model.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: fine_tuning
    owner: LargeLanguageModel
    domain_of:
    - LargeLanguageModel
    range: string
  supported_languages:
    name: supported_languages
    description: A list of languages, expressed as ISO two letter codes. For example,
      'jp, fr, en, de'
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: supported_languages
    owner: LargeLanguageModel
    domain_of:
    - LargeLanguageModel
    range: string
    multivalued: true
    inlined: true
    inlined_as_list: true
  isPartOf:
    name: isPartOf
    description: Annotation that a Large Language model is part of a family of models
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:isPartOf
    alias: isPartOf
    owner: LargeLanguageModel
    domain_of:
    - Risk
    - LargeLanguageModel
    - Stakeholder
    range: LargeLanguageModelFamily
  hasEvaluation:
    name: hasEvaluation
    description: A relationship indicating that an entity has an AI evaluation result.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: dqv:hasQualityMeasurement
    alias: hasEvaluation
    owner: LargeLanguageModel
    domain_of:
    - AiModel
    range: AiEvalResult
    multivalued: true
  architecture:
    name: architecture
    description: A description of the architecture of an AI such as 'Decoder-only'.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: architecture
    owner: LargeLanguageModel
    domain_of:
    - AiModel
    range: string
  gpu_hours:
    name: gpu_hours
    description: GPU consumption in terms of hours
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: gpu_hours
    owner: LargeLanguageModel
    domain_of:
    - AiModel
    range: integer
    minimum_value: 0
  power_consumption_w:
    name: power_consumption_w
    description: power consumption in Watts
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: power_consumption_w
    owner: LargeLanguageModel
    domain_of:
    - AiModel
    range: integer
    minimum_value: 0
  carbon_emitted:
    name: carbon_emitted
    description: The number of tons of carbon dioxide equivalent that are emitted
      during training
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: carbon_emitted
    owner: LargeLanguageModel
    domain_of:
    - AiModel
    range: float
    minimum_value: 0
    unit:
      symbol: t CO2-eq
      descriptive_name: tons of CO2 equivalent
  hasRiskControl:
    name: hasRiskControl
    description: Indicates the control measures associated with a system or component
      to modify risks.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:hasRiskControl
    alias: hasRiskControl
    owner: LargeLanguageModel
    domain_of:
    - AiModel
    range: RiskControl
    multivalued: true
  producer:
    name: producer
    description: A relationship to the Organization instance which produces this instance.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: producer
    owner: LargeLanguageModel
    domain_of:
    - BaseAi
    range: Organization
  hasModelCard:
    name: hasModelCard
    description: A relationship to model card references.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: hasModelCard
    owner: LargeLanguageModel
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
    owner: LargeLanguageModel
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
    owner: LargeLanguageModel
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
    owner: LargeLanguageModel
    domain_of:
    - BaseAi
    range: AiTask
    multivalued: true
    inlined: false
  isProvidedBy:
    name: isProvidedBy
    description: A relationship indicating the AI model has been provided by an AI
      model provider.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: airo:isProvidedBy
    alias: isProvidedBy
    owner: LargeLanguageModel
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
    owner: LargeLanguageModel
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
    owner: LargeLanguageModel
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
    owner: LargeLanguageModel
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
    owner: LargeLanguageModel
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
    owner: LargeLanguageModel
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
    owner: LargeLanguageModel
    domain_of:
    - Entity
    range: date
    required: false

```
</details>
