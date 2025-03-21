# ai-risk-ontology

An ontology describing AI systems and their risks

URI: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology

Name: ai-risk-ontology



## Classes

| Class | Description |
| --- | --- |
| [Any](Any.md) | None |
| [Container](Container.md) | An umbrella object that holds the ontology class instances |
| [Entity](Entity.md) | A generic grouping for any identifiable entity. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Action](Action.md) | Action to remediate a risk |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiEval](AiEval.md) | An AI Evaluation, e.g. a metric, benchmark, unitxt card evaluation, a question or a combination of such entities. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Question](Question.md) | An evaluation where a question has to be answered |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Questionnaire](Questionnaire.md) | A questionnaire groups questions |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiEvalResult](AiEvalResult.md) | The result of an evaluation for a specific AI model. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiLifecyclePhase](AiLifecyclePhase.md) | A Phase of AI lifecycle which indicates evolution of the system from conception through retirement. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiModelValidation](AiModelValidation.md) | AI model validation steps that have been performed after the model training to ensure high AI model quality. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[DataPreprocessing](DataPreprocessing.md) | Data transformations, such as PI filtering, performed to ensure high quality of AI model training data. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiTask](AiTask.md) | A task, such as summarization and classification, performed by an AI. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiModel](AiModel.md) | A base AI Model class. No assumption about the type (SVM, LLM, etc.). Subclassed by model types (see LargeLanguageModel). |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of language-related tasks such as generation, summarization, classification, among others. A LLM is implemented as an artificial neural networks using a transformer architecture. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities. ChatGPT is an example of an AI system which deploys multiple GPT AI models. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is capable of autonomously performing tasks on behalf of a user or another system by designing its workflow and utilizing available tools. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Dataset](Dataset.md) | A body of structured information describing some topic(s) of interest. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Documentation](Documentation.md) | Documented information about a concept or other topic(s) of interest. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[LargeLanguageModelFamily](LargeLanguageModelFamily.md) | A large language model family is a set of models that are provided by the same AI systems provider and are built around the same architecture, but differ e.g. in the number of parameters. Examples are Meta's Llama2 family or the IBM granite models. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[License](License.md) | The general notion of a license which defines terms and grants permissions to users of AI systems, datasets and software. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Modality](Modality.md) | A modality supported by an Ai component. Examples include text, image, video. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Organization](Organization.md) | Any organizational entity such as a corporation, educational institution, consortium, government, etc. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiOffice](AiOffice.md) | The EU AI Office (https://digital-strategy.ec.europa.eu/en/policies/ai-office) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiProvider](AiProvider.md) | A provider under the AI Act is defined by Article 3(3) as a natural or legal person or body that develops an AI system or general-purpose AI model or has an AI system or general-purpose AI model developed; and places that system or model on the market, or puts that system into service, under the provider's own name or trademark, whether for payment or free for charge. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential to cause harms |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[RiskConcept](RiskConcept.md) | An umbrella term for refering to risk, risk source, consequence and impact. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[RiskControl](RiskControl.md) | A measure that maintains and/or modifies risk (and risk concepts) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy. |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[RiskTaxonomy](RiskTaxonomy.md) | A taxonomy of AI system related risks |
| [Fact](Fact.md) | A fact about something, for example the result of a measurement. In addition to the value, evidence is provided. |



## Slots

| Slot | Description |
| --- | --- |
| [actions](actions.md) | A list of risk related actions |
| [aimodelfamilies](aimodelfamilies.md) | A list of AI model families |
| [aimodels](aimodels.md) | A list of AI models |
| [aitasks](aitasks.md) | A list of AI tasks |
| [architecture](architecture.md) | A description of the architecture of an AI such as 'Decoder-only' |
| [bestValue](bestValue.md) | Annotation of the best possible result of the evaluation |
| [broadMatch](broadMatch.md) | The property is used to state a hierarchical mapping link between two concept... |
| [carbon_emitted](carbon_emitted.md) | The number of tons of carbon dioxide equivalent that are emitted during train... |
| [closeMatch](closeMatch.md) | The property is used to link two concepts that are sufficiently similar that ... |
| [concern](concern.md) | Some explanation about the concern related to an AI risk |
| [contextWindowSize](contextWindowSize.md) | The total length, in bytes, of an AI model's context window |
| [datasets](datasets.md) | A list of data sets |
| [dateCreated](dateCreated.md) | The date on which the entity was created |
| [dateModified](dateModified.md) | The date on which the entity was most recently modified |
| [description](description.md) | The description of an entity |
| [descriptor](descriptor.md) | Annotates whether an AI risk is a traditional risk, specific to or amplified ... |
| [detectsRiskConcept](detectsRiskConcept.md) | The property airo:detectsRiskConcept indicates the control used for detecting... |
| [documents](documents.md) | A list of documents |
| [evaluations](evaluations.md) | A list of AI evaluation methods |
| [evidence](evidence.md) | Evidence provides a source (typical a chunk, paragraph or link) describing wh... |
| [exactMatch](exactMatch.md) | The property is used to link two concepts, indicating a high degree of confid... |
| [fine_tuning](fine_tuning.md) | A description of the fine-tuning mechanism(s) applied to a model |
| [gpu_hours](gpu_hours.md) | GPU consumption in terms of hours |
| [grants_license](grants_license.md) | A relationship from a granting entity such as an Organization to a License in... |
| [hasAiActorTask](hasAiActorTask.md) | Pertinent AI Actor Tasks for each subcategory |
| [hasDataset](hasDataset.md) | A relationship to datasets that are used |
| [hasDocumentation](hasDocumentation.md) | Indicates documentation associated with an entity |
| [hasEuAiSystemType](hasEuAiSystemType.md) | The type of system as defined by the EU AI Act |
| [hasEuRiskCategory](hasEuRiskCategory.md) | The risk category of an AI system as defined by the EU AI Act |
| [hasEvaluation](hasEvaluation.md) | A relationship indicating that an entity has an AI evaluation result |
| [hasInputModality](hasInputModality.md) | A relationship indicating the input modalities supported by an AI component |
| [hasLicense](hasLicense.md) | Indicates licenses associated with a resource |
| [hasModelCard](hasModelCard.md) | A relationship to model card references |
| [hasOutputModality](hasOutputModality.md) | A relationship indicating the output modalities supported by an AI component |
| [hasPart](hasPart.md) | A relationship where an entity has another entity |
| [hasRelatedAction](hasRelatedAction.md) | A relationship where an entity relates to an action |
| [hasRelatedRisk](hasRelatedRisk.md) | A relationship where an entity relates to a risk |
| [hasRiskControl](hasRiskControl.md) | Indicates the control measures associated with a system or component to modif... |
| [hasTrainingData](hasTrainingData.md) | A relationship indicating the datasets an AI model was trained on |
| [hasUnitxtCard](hasUnitxtCard.md) | A relationship to a Unitxt card defining the risk evaluation |
| [id](id.md) | A unique identifier to this instance of the model element |
| [isComposedOf](isComposedOf.md) | Relationship indicating the some entity is composed of other entities (includ... |
| [isDefinedByTaxonomy](isDefinedByTaxonomy.md) | A relationship where a risk or a risk group is defined by a risk taxonomy |
| [isDeployedBy](isDeployedBy.md) | A relationship indicating that an entity has been deployed by an organization |
| [isDetectedBy](isDetectedBy.md) | A relationship where a risk, risk source, consequence, or impact is detected ... |
| [isDistributedBy](isDistributedBy.md) | A relationship indicating that an entity has been distributed by an organizat... |
| [isImportedBy](isImportedBy.md) | A relationship indicating that an entity has been imported by an organization |
| [isPartOf](isPartOf.md) | A relationship where an entity is part of another entity |
| [isProvidedBy](isProvidedBy.md) | A relationship indicating the AI model has been provided by an AI model provi... |
| [isResultOf](isResultOf.md) | A relationship indicating that an entity is the result of an AI evaluation |
| [licenses](licenses.md) | A list of licenses |
| [modalities](modalities.md) | A list of AI modalities |
| [name](name.md) | A text name of this instance |
| [narrowMatch](narrowMatch.md) | The property is used to state a hierarchical mapping link between two concept... |
| [numParameters](numParameters.md) | A property indicating the number of parameters in a LLM |
| [numTrainingTokens](numTrainingTokens.md) | The number of tokens a AI model was trained on |
| [organizations](organizations.md) | A list of organizations |
| [performsTask](performsTask.md) | relationship indicating the AI tasks an AI model can perform |
| [phase](phase.md) | Annotation whether an AI risk shows specifically during the training-tuning o... |
| [power_consumption_w](power_consumption_w.md) | power consumption in Watts |
| [producer](producer.md) | A relationship to the Organization instance which produces this instance |
| [provider](provider.md) | A relationship to the Organization instance that provides this instance |
| [relatedMatch](relatedMatch.md) | The property skos:relatedMatch is used to state an associative mapping link b... |
| [riskcontrols](riskcontrols.md) | A list of AI risk controls |
| [riskgroups](riskgroups.md) | A list of AI risk groups |
| [risks](risks.md) | A list of AI risks |
| [supported_languages](supported_languages.md) | A list of languages, expressed as ISO two letter codes |
| [tag](tag.md) | A shost version of the name |
| [taxonomies](taxonomies.md) | A list of AI risk taxonomies |
| [text](text.md) | The question itself |
| [training_data_preprocessing](training_data_preprocessing.md) | relationship indicating data preprocessing steps performed on training data s... |
| [type](type.md) | Annotation whether an AI risk occurs at input or output or is non-technical |
| [url](url.md) | An optional URL associated with this instance |
| [validated_by](validated_by.md) | A relationship indicating the model validation steps after AI model training |
| [value](value.md) | Some numeric or string value |
| [version](version.md) | The version of the entity embodied by a specified resource |


## Enumerations

| Enumeration | Description |
| --- | --- |
| [AiSystemType](AiSystemType.md) |  |
| [EuAiRiskCategory](EuAiRiskCategory.md) |  |


## Types

| Type | Description |
| --- | --- |
| [Boolean](Boolean.md) | A binary (true or false) value |
| [Curie](Curie.md) | a compact URI |
| [Date](Date.md) | a date (year, month and day) in an idealized calendar |
| [DateOrDatetime](DateOrDatetime.md) | Either a date or a datetime |
| [Datetime](Datetime.md) | The combination of a date and time |
| [Decimal](Decimal.md) | A real number with arbitrary precision that conforms to the xsd:decimal speci... |
| [Double](Double.md) | A real number that conforms to the xsd:double specification |
| [Float](Float.md) | A real number that conforms to the xsd:float specification |
| [Integer](Integer.md) | An integer |
| [Jsonpath](Jsonpath.md) | A string encoding a JSON Path |
| [Jsonpointer](Jsonpointer.md) | A string encoding a JSON Pointer |
| [Ncname](Ncname.md) | Prefix part of CURIE |
| [Nodeidentifier](Nodeidentifier.md) | A URI, CURIE or BNODE that represents a node in a model |
| [Objectidentifier](Objectidentifier.md) | A URI or CURIE that represents an object in the model |
| [Sparqlpath](Sparqlpath.md) | A string encoding a SPARQL Property Path |
| [String](String.md) | A character string |
| [Time](Time.md) | A time object represents a (local) time of day, independent of any particular... |
| [Uri](Uri.md) | a complete URI |
| [Uriorcurie](Uriorcurie.md) | a URI or a CURIE |


## Subsets

| Subset | Description |
| --- | --- |
