# ai-risk-ontology

An ontology describing AI systems and their risks

URI: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology

Name: ai-risk-ontology



## Classes

| Class | Description |
| --- | --- |
| [Any](Any.md) |  |
| [Container](Container.md) | An umbrella object that holds the ontology class instances |
| [Entity](Entity.md) | A generic grouping for any identifiable entity |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiEval](AiEval.md) | An AI Evaluation, e |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Question](Question.md) | An evaluation where a question has to be answered |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Questionnaire](Questionnaire.md) | A questionnaire groups questions |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiEvalResult](AiEvalResult.md) | The result of an evaluation for a specific AI model |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiLifecyclePhase](AiLifecyclePhase.md) | A Phase of AI lifecycle which indicates evolution of the system from concepti... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiModelValidation](AiModelValidation.md) | AI model validation steps that have been performed after the model training t... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[DataPreprocessing](DataPreprocessing.md) | Data transformations, such as PI filtering, performed to ensure high quality ... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiTask](AiTask.md) | A task, such as summarization and classification, performed by an AI |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiModel](AiModel.md) | A base AI Model class |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is c... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Consequence](Consequence.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Dataset](Dataset.md) | A body of structured information describing some topic(s) of interest |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Documentation](Documentation.md) | Documented information about a concept or other topic(s) of interest |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Impact](Impact.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[IncidentStatus](IncidentStatus.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[IncidentConcludedclass](IncidentConcludedclass.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[IncidentHaltedclass](IncidentHaltedclass.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[IncidentMitigatedclass](IncidentMitigatedclass.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[IncidentNearMissclass](IncidentNearMissclass.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[IncidentOngoingclass](IncidentOngoingclass.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Input](Input.md) | Input for which the system or component generates output |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[LargeLanguageModelFamily](LargeLanguageModelFamily.md) | A large language model family is a set of models that are provided by the sam... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[License](License.md) | The general notion of a license which defines terms and grants permissions to... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Likelihood](Likelihood.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[LLMIntrinsic](LLMIntrinsic.md) | A capability that can be invoked through a well-defined API that is reasonabl... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Modality](Modality.md) | A modality supported by an Ai component |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Organization](Organization.md) | Any organizational entity such as a corporation, educational institution, con... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiOffice](AiOffice.md) | The EU AI Office (https://digital-strategy |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[AiProvider](AiProvider.md) | A provider under the AI Act is defined by Article 3(3) as a natural or legal ... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Policy](Policy.md) | A guidance document outlining any of: procedures, plans, principles, decision... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[LLMQuestionPolicy](LLMQuestionPolicy.md) | The policy guides how the language model should answer a diverse set of sensi... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Principle](Principle.md) | A representation of values or norms that must be taken into consideration whe... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[RiskConcept](RiskConcept.md) | An umbrella term for refering to risk, risk source, consequence and impact |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[RiskControl](RiskControl.md) | A measure that maintains and/or modifies risk (and risk concepts) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Action](Action.md) | Action to remediate a risk |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[RiskIncident](RiskIncident.md) | An event occuring or occured which is a realised or materialised risk |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[RiskTaxonomy](RiskTaxonomy.md) | A taxonomy of AI system related risks |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Rule](Rule.md) | A rule describing a process or control that directs or determines if and how ... |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Obligation](Obligation.md) | A rule describing an obligation for performing an activity |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Permission](Permission.md) | A rule describing a permission to perform an activity |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Prohibition](Prohibition.md) | A rule describing a prohibition to perform an activity |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Severity](Severity.md) |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Stakeholder](Stakeholder.md) | An AI system stakeholder for Responsible AI governance (e |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[StakeholderGroup](StakeholderGroup.md) | An AI system stakeholder grouping |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Term](Term.md) | A term and its definitions |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Vocabulary](Vocabulary.md) | A collection of terms, with their definitions and relationships |
| [Fact](Fact.md) | A fact about something, for example the result of a measurement |



## Slots

| Slot | Description |
| --- | --- |
| [actions](actions.md) | A list of risk related actions |
| [adapters](adapters.md) | A list of Adapters |
| [adaptsModel](adaptsModel.md) | The LargeLanguageModel being adapted |
| [aievalresults](aievalresults.md) | A list of AI evaluation results |
| [aimodelfamilies](aimodelfamilies.md) | A list of AI model families |
| [aimodels](aimodels.md) | A list of AI models |
| [aitasks](aitasks.md) | A list of AI tasks |
| [architecture](architecture.md) | A description of the architecture of an AI such as 'Decoder-only' |
| [author](author.md) | The author or authors of the documentation |
| [benchmarkmetadatacards](benchmarkmetadatacards.md) | A list of AI evaluation benchmark metadata cards |
| [bestValue](bestValue.md) | Annotation of the best possible result of the evaluation |
| [broadMatch](broadMatch.md) | The property is used to state a hierarchical mapping link between two concept... |
| [carbon_emitted](carbon_emitted.md) | The number of tons of carbon dioxide equivalent that are emitted during train... |
| [closeMatch](closeMatch.md) | The property is used to link two concepts that are sufficiently similar that ... |
| [concern](concern.md) | Some explanation about the concern related to an AI risk |
| [contextWindowSize](contextWindowSize.md) | The total length, in bytes, of an AI model's context window |
| [datasets](datasets.md) | A list of data sets |
| [dateCreated](dateCreated.md) | The date on which the entity was created |
| [dateModified](dateModified.md) | The date on which the entity was most recently modified |
| [describesAiEval](describesAiEval.md) | A relationship where a BenchmarkMetadataCard describes and AI evaluation (ben... |
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
| [hasAdapter](hasAdapter.md) | The Adapter for the intrinsic |
| [hasAdapterType](hasAdapterType.md) | The Adapter type, for example: LORA, ALORA, X-LORA |
| [hasAiActorTask](hasAiActorTask.md) | Pertinent AI Actor Tasks for each subcategory |
| [hasAnnotation](hasAnnotation.md) | The process used to annotate or label the dataset, including who or what perf... |
| [hasAudience](hasAudience.md) | The intended audience, such as researchers, developers, policymakers, etc |
| [hasBaselineResults](hasBaselineResults.md) | The results of well-known or widely used models to give context to new perfor... |
| [hasBenchmarkMetadata](hasBenchmarkMetadata.md) | A relationship to a Benchmark Metadata Card which contains metadata about the... |
| [hasCalculation](hasCalculation.md) | The way metrics are computed based on model outputs and the benchmark data |
| [hasConsequence](hasConsequence.md) | Indicates consequence(s) possible or arising from specified concept |
| [hasConsiderationComplianceWithRegulations](hasConsiderationComplianceWithRegulations.md) | Compliance with relevant legal or ethical regulations (if applicable) |
| [hasConsiderationConsentProcedures](hasConsiderationConsentProcedures.md) | Information on how consent was obtained (if applicable), especially for datas... |
| [hasConsiderationPrivacyAndAnonymity](hasConsiderationPrivacyAndAnonymity.md) | How any personal or sensitive data is handled and whether any anonymization t... |
| [hasDataFormat](hasDataFormat.md) | The structure and modality of the data (e |
| [hasDataset](hasDataset.md) | A relationship to datasets that are used |
| [hasDataSize](hasDataSize.md) | The size of the dataset, including the number of data points or examples |
| [hasDataSource](hasDataSource.md) | The origin or source of the data used in the benchmark (e |
| [hasDataType](hasDataType.md) | The type of data used in the benchmark (e |
| [hasDemographicAnalysis](hasDemographicAnalysis.md) | How the benchmark evaluates performance across different demographic groups (... |
| [hasDocumentation](hasDocumentation.md) | Indicates documentation associated with an entity |
| [hasDomains](hasDomains.md) | The specific domains or areas where the benchmark is applied (e |
| [hasEuAiSystemType](hasEuAiSystemType.md) | The type of system as defined by the EU AI Act |
| [hasEuRiskCategory](hasEuRiskCategory.md) | The risk category of an AI system as defined by the EU AI Act |
| [hasEvaluation](hasEvaluation.md) | A relationship indicating that an entity has an AI evaluation result |
| [hasException](hasException.md) | Exception type |
| [hasGoal](hasGoal.md) | The specific goal or primary use case the benchmark is designed for |
| [hasImpact](hasImpact.md) | Indicates impact(s) possible or arising as consequences from specified concep... |
| [hasImpactOn](hasImpactOn.md) | Indicates impact(s) possible or arising as consequences from specified concep... |
| [hasImplementation](hasImplementation.md) | A relationship to a implementation defining the risk evaluation |
| [hasInputModality](hasInputModality.md) | A relationship indicating the input modalities supported by an AI component |
| [hasInterpretation](hasInterpretation.md) | How users should interpret the scores or results from the metrics |
| [hasLanguages](hasLanguages.md) | The languages included in the dataset used by the benchmark (e |
| [hasLicense](hasLicense.md) | Indicates licenses associated with a resource |
| [hasLikelihood](hasLikelihood.md) | The likelihood or probability or chance of something taking place or occuring |
| [hasLimitations](hasLimitations.md) | Limitations in evaluating or addressing risks, such as gaps in demographic co... |
| [hasMethods](hasMethods.md) | The evaluation techniques applied within the benchmark |
| [hasMetrics](hasMetrics.md) | The specific performance metrics used to assess models (e |
| [hasModelCard](hasModelCard.md) | A relationship to model card references |
| [hasOutOfScopeUses](hasOutOfScopeUses.md) | Use cases where the benchmark is not designed to be applied and could give mi... |
| [hasOutputModality](hasOutputModality.md) | A relationship indicating the output modalities supported by an AI component |
| [hasParentDefinition](hasParentDefinition.md) | Indicates parent terms associated with a term |
| [hasPart](hasPart.md) | A relationship where an entity has another entity |
| [hasReasonDenial](hasReasonDenial.md) | Reason for denial |
| [hasRelatedAction](hasRelatedAction.md) | A relationship where an entity relates to an action |
| [hasRelatedRisk](hasRelatedRisk.md) | A relationship where an entity relates to a risk |
| [hasRelatedTerm](hasRelatedTerm.md) | A relationship where an entity relates to a term |
| [hasResources](hasResources.md) | Links to relevant resources, such as repositories or papers related to the be... |
| [hasRiskControl](hasRiskControl.md) | Indicates the control measures associated with a system or component to modif... |
| [hasRule](hasRule.md) | Specifying applicability or inclusion of a rule within specified context |
| [hasSeverity](hasSeverity.md) | Indicates the severity associated with a concept |
| [hasShortReplyType](hasShortReplyType.md) | Short reply type |
| [hasSimilarBenchmarks](hasSimilarBenchmarks.md) | Benchmarks that are closely related in terms of goals or data type |
| [hasStatus](hasStatus.md) | Indicates the status of specified concept |
| [hasSubDefinition](hasSubDefinition.md) | Indicates child terms associated with a term |
| [hasTasks](hasTasks.md) | The tasks or evaluations the benchmark is intended to assess |
| [hasTerm](hasTerm.md) | Indicates terms associated with a vocabulary |
| [hasTrainingData](hasTrainingData.md) | A relationship indicating the datasets an AI model was trained on |
| [hasUnitxtCard](hasUnitxtCard.md) | A relationship to a Unitxt card defining the risk evaluation |
| [hasValidation](hasValidation.md) | Measures taken to ensure that the benchmark provides valid and reliable evalu... |
| [hasVariant](hasVariant.md) | Indicates an incident that shares the same causative factors, produces simila... |
| [id](id.md) | A unique identifier to this instance of the model element |
| [isComposedOf](isComposedOf.md) | Relationship indicating the some entity is composed of other entities (includ... |
| [isDefinedByTaxonomy](isDefinedByTaxonomy.md) | A relationship where a risk or a risk group is defined by a risk taxonomy |
| [isDefinedByVocabulary](isDefinedByVocabulary.md) | A relationship where a term or a term group is defined by a vocabulary |
| [isDeployedBy](isDeployedBy.md) | A relationship indicating that an entity has been deployed by an organization |
| [isDetectedBy](isDetectedBy.md) | A relationship where a risk, risk source, consequence, or impact is detected ... |
| [isDistributedBy](isDistributedBy.md) | A relationship indicating that an entity has been distributed by an organizat... |
| [isImportedBy](isImportedBy.md) | A relationship indicating that an entity has been imported by an organization |
| [isPartOf](isPartOf.md) | A relationship where an entity is part of another entity |
| [isProvidedBy](isProvidedBy.md) | A relationship indicating the AI model has been provided by an AI model provi... |
| [isResultOf](isResultOf.md) | A relationship indicating that an entity is the result of an AI evaluation |
| [licenses](licenses.md) | A list of licenses |
| [llmintrinsics](llmintrinsics.md) | A list of LLMIntrinsics |
| [llmquestionpolicies](llmquestionpolicies.md) | A list of LLM question policies |
| [modalities](modalities.md) | A list of AI modalities |
| [name](name.md) | A text name of this instance |
| [narrowMatch](narrowMatch.md) | The property is used to state a hierarchical mapping link between two concept... |
| [numParameters](numParameters.md) | A property indicating the number of parameters in a LLM |
| [numTrainingTokens](numTrainingTokens.md) | The number of tokens a AI model was trained on |
| [obligations](obligations.md) | A list of Obligations |
| [organizations](organizations.md) | A list of organizations |
| [overview](overview.md) | A brief description of the benchmark's main goals and scope |
| [performsTask](performsTask.md) | relationship indicating the AI tasks an AI model can perform |
| [permissions](permissions.md) | A list of Permissions |
| [phase](phase.md) | Annotation whether an AI risk shows specifically during the training-tuning o... |
| [policies](policies.md) | A list of policies |
| [power_consumption_w](power_consumption_w.md) | power consumption in Watts |
| [principles](principles.md) | A list of Principles |
| [producer](producer.md) | A relationship to the Organization instance which produces this instance |
| [prohibitions](prohibitions.md) | A list of prohibitions |
| [provider](provider.md) | A relationship to the Organization instance that provides this instance |
| [refersToRisk](refersToRisk.md) | Indicates the incident (subject) is a materialisation of the indicated risk (... |
| [relatedMatch](relatedMatch.md) | The property skos:relatedMatch is used to state an associative mapping link b... |
| [riskcontrols](riskcontrols.md) | A list of AI risk controls |
| [riskgroups](riskgroups.md) | A list of AI risk groups |
| [riskincidents](riskincidents.md) | A list of AI risk incidents |
| [risks](risks.md) | A list of AI risks |
| [rules](rules.md) | A list of rules |
| [source_uri](source_uri.md) | The uri of the incident |
| [stakeholdergroups](stakeholdergroups.md) | A list of AI stakeholder groups |
| [stakeholders](stakeholders.md) | A list of stakeholders |
| [supported_languages](supported_languages.md) | A list of languages, expressed as ISO two letter codes |
| [tag](tag.md) | A shost version of the name |
| [taxonomies](taxonomies.md) | A list of AI risk taxonomies |
| [terms](terms.md) | A list of terms from a vocabulary |
| [text](text.md) | The question itself |
| [training_data_preprocessing](training_data_preprocessing.md) | relationship indicating data preprocessing steps performed on training data s... |
| [type](type.md) | Annotation whether an AI risk occurs at input or output or is non-technical |
| [url](url.md) | An optional URL associated with this instance |
| [validated_by](validated_by.md) | A relationship indicating the model validation steps after AI model training |
| [value](value.md) | Some numeric or string value |
| [version](version.md) | The version of the entity embodied by a specified resource |
| [vocabularies](vocabularies.md) | A list of vocabularies |


## Enumerations

| Enumeration | Description |
| --- | --- |
| [AdapterType](AdapterType.md) |  |
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
