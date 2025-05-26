LIST_OF_STR_SCHEMA = {
    "type": "array",
    "items": {"enum": None},
}

QUESTIONNAIRE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": [
        "answer",
        "explanation",
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
            "type": "array",
            "items": {
                "enum": [
                    "Unacceptable Risk",
                    "High Risk",
                    "Limited Risk",
                    "Minimal Risk",
                ]
            },
        },
        "Relevant Text from the EU AI Act": {},
        "Reasoning": {"type": "string"},
    },
    "required": [
        "Description",
        "Classification",
        "Relevant Text from the EU AI Act",
        "Reasoning",
    ],
}
