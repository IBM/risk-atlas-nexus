from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import (
    EuAiRiskCategory,
)


LIST_OF_STR_SCHEMA = {
    "type": "array",
    "items": {"enum": None},
}

QUESTIONNAIRE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "explanation": {"type": "string"},
        "confidence": {"type": "string"},
    },
    "required": [
        "answer",
        "explanation",
        "confidence"
    ],
}

DOMAIN_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": [
                "Customer service/support",
                "Technical",
                "Information retrieval",
                "Strategy",
                "Code/software engineering",
                "Communications",
                "IT/business automation",
                "Writing assistant",
                "Financial",
                "Talent and Organization including HR",
                "Product",
                "Marketing",
                "Cybersecurity",
                "Healthcare",
                "User Research",
                "Sales",
                "Risk and Compliance",
                "Design",
                "Other",
            ],
        },
        "explanation": {"type": "string"},
    },
    "required": [
        "answer",
        "explanation",
    ],
}

RISK_CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "Description": {"type": "string"},
        "Classification": {
            "type": "string",
            "enum": [
                EuAiRiskCategory.EXCLUDED.value,
                EuAiRiskCategory.PROHIBITED.value,
                EuAiRiskCategory.HIGH_RISK_EXCEPTION.value,
                EuAiRiskCategory.HIGH_RISK.value,
                EuAiRiskCategory.LIMITED_OR_LOW_RISK.value,
            ],
        },
        "AIActText": {"type": "string"},
        "Reasoning": {"type": "string"},
    },
    "required": [
        "Description",
        "Classification",
        "AIActText",
        "Reasoning",
    ],
}
