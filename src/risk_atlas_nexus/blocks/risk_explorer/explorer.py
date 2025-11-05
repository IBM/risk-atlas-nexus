from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus.blocks.risk_explorer.base import ExplorerBase
from risk_atlas_nexus.data.knowledge_graph import *


class RiskExplorer(ExplorerBase):

    def __init__(self, ontology):

        # load the data into the graph
        self._ontology = ontology
        self._risks = ontology.risks or []
        self._riskcontrols = ontology.riskcontrols or []
        self._riskincidents = ontology.riskincidents or []
        self._actions = ontology.actions or []
        self._taxonomies = ontology.taxonomies or []
        self._evaluations = ontology.evaluations or []
        self._aievaluationresults = ontology.aievalresults or []
        self._benchmarkmetadatacards = ontology.benchmarkmetadatacards or []
        self._documents = ontology.documents or []
        self._datasets = ontology.datasets or []
        self._stakeholders = ontology.stakeholders or []
        self._llmintrinsics = ontology.llmintrinsics or []
        self._adapters = ontology.adapters or []
        self._llmquestionpolicies = ontology.llmquestionpolicies or []
        self._rules = ontology.rules or []
        self._principles = ontology.principles or []

    def get_all_risks(self, taxonomy=None):
        """Get all risk definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Risk]
                Result containing a list of AI risks
        """
        risk_instances = self._risks
        if taxonomy is not None:
            risk_instances = list(
                filter(
                    lambda risk: risk.isDefinedByTaxonomy == taxonomy, risk_instances
                )
            )

        return risk_instances

    def get_risk(self, tag=None, id=None, name=None, taxonomy=None):
        """Get a risk definition from the LinkML by tag, id or name

        Args:
            id: (Optional) str
                The string ID identifying the risk
            tag: (Optional) str
                The string tag identifying the risk
            name: (Optional) str
                The string name identifying the risk
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            Risk
                An AI risk result
        """
        matching_risks = self._risks
        if tag is not None:
            matching_risks = list(filter(lambda risk: risk.tag == tag, matching_risks))
        if id is not None:
            matching_risks = list(filter(lambda risk: risk.id == id, matching_risks))
        if name is not None:
            matching_risks = list(
                filter(lambda risk: risk.name == name, matching_risks)
            )
        if taxonomy is not None:
            matching_risks = list(
                filter(
                    lambda risk: risk.isDefinedByTaxonomy == taxonomy, matching_risks
                )
            )

        if len(matching_risks) > 0:
            return matching_risks[0]
        else:
            print("No matching risk found")
            return None

    def _combine_related_risks(self, initial_risk):
        """Convenience function to combine the related risks from a Risk

        Args:
            initial_risk: Risk
                The risk to parse for related risks

        Returns:
            list[str]
                Result containing a list of AI risk IDs
        """
        related_risks = []
        if initial_risk.broadMatch is not None:
            related_risks.append(initial_risk.broadMatch)
        if initial_risk.closeMatch is not None:
            related_risks.append(initial_risk.closeMatch)
        if initial_risk.exactMatch is not None:
            related_risks.append(initial_risk.exactMatch)
        if initial_risk.narrowMatch is not None:
            related_risks.append(initial_risk.narrowMatch)
        if initial_risk.relatedMatch is not None:
            related_risks.append(initial_risk.relatedMatch)

        related_risk_ids = [j for i in related_risks for j in i]
        related_risks = list(
            filter(lambda risk: risk.id in related_risk_ids, self._risks)
        )
        return related_risks

    def get_related_risks(self, risk=None, tag=None, id=None, name=None, taxonomy=None):
        """Get all related risk definitions from the LinkML by the IBM risk atlas tag

        Args:
            risk: (Optional) Risk
                The Risk object to find related risks for
            id: (Optional) str
                The string ID identifying the risk
            tag: (Optional) str
                The string tag identifying the risk
            name: (Optional) str
                The string name identifying the risk
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[str]
                Result containing a list of AI risks IDs
        """
        matching_risks = self._risks

        if risk is not None:
            matching_risks = [risk]
        if tag is not None:
            matching_risks = list(filter(lambda risk: risk.tag == tag, matching_risks))
        if id is not None:
            matching_risks = list(filter(lambda risk: risk.id == id, matching_risks))
        if name is not None:
            matching_risks = list(
                filter(lambda risk: risk.name == name, matching_risks)
            )

        if len(matching_risks) > 0:
            initial_risk: Risk = matching_risks[0]
            related_risks = self._combine_related_risks(initial_risk)

            if taxonomy is not None:
                related_risks = list(
                    filter(
                        lambda risk: risk.isDefinedByTaxonomy == taxonomy, related_risks
                    )
                )

            return related_risks
        else:
            print("No matching risk found")
            return None

    def get_related_actions(
        self, risk=None, tag=None, id=None, name=None, taxonomy=None
    ):
        """Get actions for a risk from the LinkML

        Args:
            risk: (Optional) Risk
                The Risk object to find related actions for
            id: (Optional) str
                The string ID identifying the risk to find related actions for
            tag: (Optional) str
                The string tag identifying the risk to find related actions for
            name: (Optional) str
                The string name identifying the risk to find related actions for
            taxonomy: str
                (Optional) Only return actions which come from this taxonomy

        Returns:
            list[Action]
                Result containing a list of the actions which are marked as related to the specified AI risk
        """
        matching_risks = self._risks

        if risk is not None:
            matching_risks = [risk]
        if tag is not None:
            matching_risks = list(filter(lambda risk: risk.tag == tag, matching_risks))
        if id is not None:
            matching_risks = list(filter(lambda risk: risk.id == id, matching_risks))
        if name is not None:
            matching_risks = list(
                filter(lambda risk: risk.name == name, matching_risks)
            )

        if len(matching_risks) > 0:
            risk: Risk = matching_risks[0]
            actions = []

            if risk.hasRelatedAction is not None:
                actions.append(risk.hasRelatedAction)

            actions = [j for i in actions for j in i]

            if taxonomy is not None:
                actions = list(
                    filter(
                        lambda action: action.isDefinedByTaxonomy == taxonomy, actions
                    )
                )

            return actions
        else:
            print("No matching actions found")
            return None

    def get_all_actions(self, taxonomy=None):
        """Get all actions definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Action]
                Result containing a list of AI actions
        """
        action_instances = self._actions

        if taxonomy is not None:
            action_instances = list(
                filter(
                    lambda action: action.isDefinedByTaxonomy == taxonomy,
                    action_instances,
                )
            )

        return action_instances

    def get_action_by_id(self, id):
        """Get action definition from the LinkML by ID

        Args:
            id: str
                The string id for an action

        Returns:
            Action
                Result containing an AI actions
        """
        matching_actions = list(filter(lambda action: action.id == id, self._actions))

        if len(matching_actions) > 0:
            return matching_actions[0]
        else:
            print("No matching risk found")
            return None

    def get_all_taxonomies(self):
        """Get all taxonomy definitions from the LinkML

        Returns:
            list[RiskTaxonomy]
                Result containing a list of risk taxonomies
        """
        return self._taxonomies

    def get_taxonomy_by_id(self, id):
        """Get taxonomy definition from the LinkML by ID

        Args:
            id: str
                The string id for a taxonomy

        Returns:
            RiskTaxonomy
                Result containing an AI RiskTaxonomy
        """
        matching_taxonomies = list(
            filter(lambda taxonomy: taxonomy.id == id, self._taxonomies)
        )

        if len(matching_taxonomies) > 0:
            return matching_taxonomies[0]
        else:
            print("No matching taxonomy found")
            return None

    def get_related_risk_controls(
        self, risk=None, tag=None, id=None, name=None, taxonomy=None
    ):
        """Get related risk controls for a risk from the LinkML

        Args:
            risk: (Optional) Risk
                The Risk object to find related actions for
            id: (Optional) str
                The string ID identifying the risk to find related actions for
            tag: (Optional) str
                The string tag identifying the risk to find related actions for
            name: (Optional) str
                The string name identifying the risk to find related actions for
            taxonomy: str
                (Optional) The string label for a taxonomy, to filter action results by

        Returns:
            list[RiskControl]
                Result containing a list of the risk controls which are marked as related to the specified AI risk
        """
        matching_risks = self._risks

        if risk is not None:
            matching_risks = [risk]
        if tag is not None:
            matching_risks = list(filter(lambda risk: risk.tag == tag, matching_risks))
        if id is not None:
            matching_risks = list(filter(lambda risk: risk.id == id, matching_risks))
        if name is not None:
            matching_risks = list(
                filter(lambda risk: risk.name == name, matching_risks)
            )

        if len(matching_risks) > 0:
            risk: Risk = matching_risks[0]
            risk_controls = []

            if risk.isDetectedBy is not None:
                risk_controls.append(risk.isDetectedBy)

            risk_controls = [j for i in risk_controls for j in i]

            if taxonomy is not None:
                risk_controls = list(
                    filter(
                        lambda risk_control: risk_control.isDefinedByTaxonomy
                        == taxonomy,
                        risk_controls,
                    )
                )
            related_risk_controls = list(
                filter(
                    lambda risk_control: risk_control.id in risk_controls,
                    self._riskcontrols,
                )
            )
            return related_risk_controls
        else:
            print("No matching risk controls found")
            return None

    def get_all_risk_controls(self, taxonomy=None):
        """Get all risk control definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[RiskControl]
                Result containing a list of RiskControls
        """
        risk_control_instances = self._riskcontrols

        if taxonomy is not None:
            risk_control_instances = list(
                filter(
                    lambda risk_control: risk_control.isDefinedByTaxonomy == taxonomy,
                    risk_control_instances,
                )
            )

        return risk_control_instances

    def get_risk_control(self, id):
        """Get risk control definition from the LinkML by ID

        Args:
            id: str
                The string id for a risk control

        Returns:
            RiskControl
                Result containing a RiskControl
        """
        matching_risk_controls = list(
            filter(lambda risk_control: risk_control.id == id, self._riskcontrols)
        )

        if len(matching_risk_controls) > 0:
            return matching_risk_controls[0]
        else:
            print("No matching risk control found")
            return None

    def get_risk_incidents(self, taxonomy=None):
        """Get risk incident instances

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[RiskIncident]
                Result containing a list of RiskIncidents
        """
        risk_incident_instances = self._riskincidents or []

        if taxonomy is not None:
            risk_incident_instances = list(
                filter(
                    lambda risk_incident: risk_incident.isDefinedByTaxonomy == taxonomy,
                    risk_incident_instances,
                )
            )

        return risk_incident_instances

    def get_risk_incident(self, id):
        """Get risk incident instance by ID

        Args:
            id: str
                The string id for a risk incident

        Returns:
            RiskIncident
                Result containing a RiskIncident
        """
        risk_incident_instances = self._riskincidents or []

        matching_risk_incidents = list(
            filter(lambda risk_control: risk_control.id == id, risk_incident_instances)
        )

        if len(matching_risk_incidents) > 0:
            return matching_risk_incidents[0]
        else:
            print("No matching risk incident found")
            return None

    def get_related_risk_incidents(self, risk=None, risk_id=None, taxonomy=None):
        """Get related risk incidents for a risk

        Args:
            risk: (Optional) Risk
                The Risk object to find related incidents for
            risk_id: (Optional) str
            taxonomy: str
                (Optional) The string label for a taxonomy, to filter action results by

        Returns:
            list[RiskIncident]
                Result containing a list of the risk incidents which are marked as related to the specified AI risk
        """
        matching_risks = self._risks

        if risk is not None:
            matching_risks = [risk]
        if risk_id is not None:
            matching_risks = list(
                filter(lambda risk: risk.id == risk_id, matching_risks)
            )

        if len(matching_risks) > 0:
            risk: Risk = matching_risks[0]

            risk_incident_instances = self._riskincidents or []

            risk_incident_instances = list(
                filter(
                    lambda risk_incident: risk.id in risk_incident.refersToRisk,
                    risk_incident_instances,
                )
            )
            if taxonomy is not None:
                risk_incident_instances = list(
                    filter(
                        lambda risk_incident: risk_incident.isDefinedByTaxonomy
                        == taxonomy,
                        risk_incident_instances,
                    )
                )

            return risk_incident_instances
        else:
            print("No matching risk incidents found")
            return None

    def get_all_evaluations(self, taxonomy=None):
        """Get all evaluation definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[AiEval]
                Result containing a list of AiEval
        """
        evaluation_instances = self._evaluations

        if taxonomy is not None:
            evaluation_instances = list(
                filter(
                    lambda evaluation: evaluation.isDefinedByTaxonomy == taxonomy,
                    evaluation_instances,
                )
            )

        return evaluation_instances

    def get_evaluation(self, id):
        """Get evaluation definition from the LinkML by ID

        Args:
            id: str
                The string id for an evaluation

        Returns:
            AiEval
                Result containing an AiEval
        """
        matching_evaluations = list(
            filter(lambda evaluation: evaluation.id == id, self._evaluations)
        )

        if len(matching_evaluations) > 0:
            return matching_evaluations[0]
        else:
            print("No matching evaluation found")
            return None

    def get_related_evaluations(self, risk=None, risk_id=None, taxonomy=None):
        """Get related evaluations for a risk

        Args:
            risk: (Optional) Risk
                The Risk object to find related evaluations for
            risk_id: (Optional) str
            taxonomy: str
                (Optional) The string label for a taxonomy, to filter action results by

        Returns:
            list[AiEval]
                Result containing a list of the evaluations which are marked as related to the specified AI risk
        """
        matching_risks = self._risks

        if risk is not None:
            matching_risks = [risk]
        if risk_id is not None:
            matching_risks = list(
                filter(lambda risk: risk.id == risk_id, matching_risks)
            )

        if len(matching_risks) > 0:
            risk: Risk = matching_risks[0]

            evaluation_instances = self._evaluations or []

            evaluation_instances = list(
                filter(
                    lambda evaluation: risk.id in (evaluation.hasRelatedRisk or []),
                    evaluation_instances,
                )
            )
            if taxonomy is not None:
                evaluation_instances = list(
                    filter(
                        lambda evaluation: evaluation.isDefinedByTaxonomy == taxonomy,
                        evaluation_instances,
                    )
                )

            return evaluation_instances
        else:
            print("No matching evaluations found")
            return []

    def get_all_benchmark_metadata_cards(self, taxonomy=None, aieval_id=None):
        """Get all benchmark metadata definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy
            aieval_id: str
                (Optional) The string id for an AiEval that the benchmark describes

        Returns:
            list[AiEval]
                Result containing a list of BenchmarkMetadataCards
        """
        benchmark_metadata_card_instances = self._benchmarkmetadatacards or []

        if taxonomy is not None:
            benchmark_metadata_card_instances = list(
                filter(
                    lambda benchmark_metadata_card: benchmark_metadata_card.isDefinedByTaxonomy
                    == taxonomy,
                    benchmark_metadata_card_instances,
                )
            )

        if aieval_id is not None:
            benchmark_metadata_card_instances = list(
                filter(
                    lambda benchmark_metadata_card: benchmark_metadata_card.describesAiEval
                    == aieval_id,
                    benchmark_metadata_card_instances,
                )
            )

        return benchmark_metadata_card_instances

    def get_benchmark_metadata_card(self, id):
        """Get benchmark_metadata_card definition from the LinkML by ID

        Args:
            id: str
                The string id for a benchmark_metadata_card

        Returns:
            BenchmarkMetadataCard
                Result containing an BenchmarkMetadataCard
        """
        matching_benchmark_metadata_cards = list(
            filter(
                lambda benchmark_metadata_card: benchmark_metadata_card.id == id,
                self._benchmarkmetadatacards,
            )
        )

        if len(matching_benchmark_metadata_cards) > 0:
            return matching_benchmark_metadata_cards[0]
        else:
            print("No matching benchmark_metadata_card found")
            return None

    def get_documents(self, taxonomy=None):
        """Get all documentation definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Documentation]
                Result containing a list of Documentation entries
        """
        documentation_instances = self._documents or []

        if taxonomy is not None:
            documentation_instances = list(
                filter(
                    lambda document: document.isDefinedByTaxonomy == taxonomy,
                    documentation_instances,
                )
            )

        return documentation_instances

    def get_document(self, id):
        """Get document definition from the LinkML by ID

        Args:
            id: str
                The string id for a documentation entry

        Returns:
            Documentation
                Result containing a Documentation
        """
        matching_documentation_instances = list(
            filter(lambda document: document.id == id, self._documents)
        )

        if len(matching_documentation_instances) > 0:
            return matching_documentation_instances[0]
        else:
            print("No matching documents found")
            return None

    def get_datasets(self, taxonomy=None):
        """Get all dataset definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Dataset]
                Result containing a list of Dataset entries
        """
        dataset_instances = self._datasets or []

        if taxonomy is not None:
            dataset_instances = list(
                filter(
                    lambda dataset: dataset.isDefinedByTaxonomy == taxonomy,
                    dataset_instances,
                )
            )

        return dataset_instances

    def get_dataset(self, id):
        """Get dataset definition from the LinkML by ID

        Args:
            id: str
                The string id for a dataset entry

        Returns:
            Dataset
                Result containing a Dataset
        """
        matching_dataset_instances = list(
            filter(lambda dataset: dataset.id == id, self._datasets)
        )

        if len(matching_dataset_instances) > 0:
            return matching_dataset_instances[0]
        else:
            print("No matching dataset found")
            return []

    def get_stakeholders(self, taxonomy=None):
        """Get all stakeholder definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Stakeholder]
                Result containing a list of Stakeholder entries
        """
        stakeholder_instances = self._stakeholders or []

        if taxonomy is not None:
            stakeholder_instances = list(
                filter(
                    lambda stakeholder: stakeholder.isDefinedByTaxonomy == taxonomy,
                    stakeholder_instances,
                )
            )

        return stakeholder_instances

    def get_stakeholder(self, id):
        """Get Stakeholder definition from the LinkML by ID

        Args:
            id: str
                The string id for a Stakeholder entry

        Returns:
            Stakeholder
                Result containing a Stakeholder
        """
        matching_stakeholder_instances = list(
            filter(lambda stakeholder: stakeholder.id == id, self._stakeholders)
        )

        if len(matching_stakeholder_instances) > 0:
            return matching_stakeholder_instances[0]
        else:
            print("No matching stakeholder found")
            return []


    def get_llmintrinsics(self, taxonomy=None):
        """Get all LLMIntrinsic definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[LLMIntrinsic]
                Result containing a list of LLMIntrinsic entries
        """
        llmintrinsic_instances = self._llmintrinsics or []

        if taxonomy is not None:
            llmintrinsic_instances = list(
                filter(
                    lambda llmintrinsic: llmintrinsic.isDefinedByTaxonomy == taxonomy,
                    llmintrinsic_instances,
                )
            )

        return llmintrinsic_instances

    def get_llmintrinsic(self, id):
        """Get LLMIntrinsic definition from the LinkML by ID

        Args:
            id: str
                The string id for a LLMIntrinsic entry

        Returns:
            LLMIntrinsic
                Result containing a LLMIntrinsic
        """
        matching_llmintrinsic_instances = list(
            filter(lambda llmintrinsic: llmintrinsic.id == id, self._llmintrinsics)
        )

        if len(matching_llmintrinsic_instances) > 0:
            return matching_llmintrinsic_instances[0]
        else:
            print("No matching LLMIntrinsic found")
            return []

    def get_related_llmintrinsics(
        self, risk=None, tag=None, risk_id=None, name=None, taxonomy=None
    ):
        """Get llmintrinsics for a risk from the LinkML

        Args:
            risk: (Optional) Risk
                The Risk object to find related llmintrinsics for
            risk_id: (Optional) str
                The string ID identifying the risk to find related llmintrinsics for
            tag: (Optional) str
                The string tag identifying the risk to find related llmintrinsics for
            name: (Optional) str
                The string name identifying the risk to find related llmintrinsics for
            taxonomy: str
                (Optional) Only return llmintrinsics which come from this taxonomy

        Returns:
            list[LLMIntrinsics]
                Result containing a list of the llmintrinsics which are marked as related to the specified AI risk
        """
        matching_risks = self._risks
        llmintrinsic_instances = []

        if risk is not None:
            matching_risks = [risk]
        if tag is not None:
            matching_risks = list(filter(lambda risk: risk.tag == tag, matching_risks))
        if risk_id is not None:
            matching_risks = list(filter(lambda risk: risk.id == risk_id, matching_risks))
        if name is not None:
            matching_risks = list(
                filter(lambda risk: risk.name == name, matching_risks)
            )

        if len(matching_risks) > 0:
            risk: Risk = matching_risks[0]
            llmintrinsics = [x for x in self._llmintrinsics if (x.hasRelatedRisk is not None and risk.id in x.hasRelatedRisk)]
            if taxonomy is not None:
                llmintrinsic_instances = list(
                    filter(
                        lambda llmintrinsic: llmintrinsic.isDefinedByTaxonomy == taxonomy,
                        llmintrinsics,
                    )
                )


            return llmintrinsic_instances
        else:
            print("No matching intrinsics found")
            return []

    def get_adapters(self, taxonomy=None):
        """Get all Adapter definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[Adapter]
                Result containing a list of Adapter entries
        """
        adapter_instances = self._adapters or []

        if taxonomy is not None:
            adapter_instances = list(
                filter(
                    lambda adapter: adapter.isDefinedByTaxonomy == taxonomy,
                    adapter_instances,
                )
            )

        return adapter_instances

    def get_adapter(self, id):
        """Get Adapter definition from the LinkML by ID

        Args:
            id: str
                The string id for a Adapter entry

        Returns:
            Adapter
                Result containing an Adapter
        """
        matching_adapter_instances = list(
            filter(lambda adapter: adapter.id == id, self._adapters)
        )

        if len(matching_adapter_instances) > 0:
            return matching_adapter_instances[0]
        else:
            print("No matching adapter found")
            return []

    def get_llm_question_policies(self, taxonomy=None):
        """Get all LLM Question Policy definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy

        Returns:
            list[LLMQuestionPolicy]
                Result containing a list of LLM Question Policy entries
        """
        llm_question_policy_instances = self._llmquestionpolicies or []

        if taxonomy is not None:
            llm_question_policy_instances = list(
                filter(
                    lambda llmqp: llmqp.isDefinedByTaxonomy == taxonomy,
                    llm_question_policy_instances,
                )
            )

        return llm_question_policy_instances

    def get_llm_question_policy(self, id):
        """Get LLM Question Policy definition from the LinkML by ID

        Args:
            id: str
                The string id for a LLM Question Policy entry

        Returns:
            policy
                Result containing a LLM Question Policy
        """
        matching_llm_question_policy_instances = list(
            filter(lambda llm_question_policy: llm_question_policy.id == id, self._llmquestionpolicies)
        )

        if len(matching_llm_question_policy_instances) > 0:
            return matching_llm_question_policy_instances[0]
        else:
            print("No matching policy found")
            return []

    def get_principles(self, taxonomy=None, document=None):
        """Get all Principle definitions from the LinkML

        Args:
            taxonomy: str
                (Optional) The string label for a taxonomy
            document: str
                (Optional) The string label for a document

        Returns:
            list[Principle]
                Result containing a list of Principle entries
        """
        principle_instances = self._principles or []

        if taxonomy is not None:
            principle_instances = list(
                filter(
                    lambda principle: principle.isDefinedByTaxonomy == taxonomy,
                    principle_instances,
                )
            )
        if document is not None:
            principle_instances = list(
                filter(
                    lambda principle: principle.hasDocumentation == document,
                    principle_instances,
                )
            )

        return principle_instances

    def get_principle(self, id):
        """Get Principle definition from the LinkML by ID

        Args:
            id: str
                The string id for a Principle entry

        Returns:
            Principle
                Result containing a Principle
        """
        matching_principle_instances = list(
            filter(lambda principle: principle.id == id, self._principles)
        )

        if len(matching_principle_instances) > 0:
            return matching_principle_instances[0]
        else:
            print("No matching principle found")
            return []
