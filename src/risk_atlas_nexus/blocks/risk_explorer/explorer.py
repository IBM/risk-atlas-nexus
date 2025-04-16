from risk_atlas_nexus.blocks.risk_explorer.base import ExplorerBase
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus.data.knowledge_graph import *


class RiskExplorer(ExplorerBase):

    def __init__(self, ontology):

        # load the data into the graph
        self._ontology = ontology
        self._risks = ontology.risks
        self._riskcontrols = ontology.riskcontrols
        self._riskincidents = ontology.riskincidents
        self._actions = ontology.actions
        self._taxonomies = ontology.taxonomies

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
        
    def get_related_risk_incidents(
        self, risk=None, risk_id=None, taxonomy=None
    ):
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
        if id is not None:
            matching_risks = list(filter(lambda risk: risk.id == risk_id, matching_risks))

        if len(matching_risks) > 0:
            risk: Risk = matching_risks[0]
            risk_incidents = []

            if risk.isDetectedBy is not None:
                risk_incidents.append(risk.isDetectedBy)

            risk_incidents = [j for i in risk_incidents for j in i]

            if taxonomy is not None:
                risk_incidents = list(
                    filter(
                        lambda risk_incident: risk_incident.isDefinedByTaxonomy
                        == taxonomy,
                        risk_incidents,
                    )
                )
            related_risk_incidents = list(
                filter(
                    lambda risk_incident: risk_incident.id in risk_incidents,
                    self._riskcontrols,
                )
            )
            return related_risk_incidents
        else:
            print("No matching risk controls found")
            return None