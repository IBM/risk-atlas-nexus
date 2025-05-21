QUESTIONNAIRE_COT_TEMPLATE = """
        I want you to play the role of a compliance officer and answer the question based on the given Intent.
        Return the question, answer and explanation in a json format where question, answer and explanation are keys of the json exactly as shown in the examples.
        you should answer the question followed by an explanation on how that answer was generated.
{% if data.examples is not none %}{% for example in data.examples %}
        Intent: {{ example.intent }}
        Question: {{ data.question }}
        Answer: {{ example.answer }}
{% if example.explanation is not none %}        Explanation: {{ example.explanation }}{% endif %}
{% endfor %}{% endif %}
        Intent: {{ usecase }}
        Question: {{ data.question }}
"""

RISK_IDENTIFICATION_TEMPLATE = """You are an expert at AI risk classification. Study the risks JSON below containing list of risk category and its description. 

risks:
{{ risks }}
The task is to identify the potential risks associated with the Input context and classify it into the given risk categories. If input doesn't fit into any of the above categories, classify it as Unknown. Respond with a  list of attribute 'category' containing the classification labels.

Examples:
{% for example in data.examples %}
Input: {{ example.Input }}
Output: {{ example.Output }}
{% endfor %}
Input: {{ usecase }}
Output: """

AI_TASKS_TEMPLATE = """Study and understand the JSON below containing a list of LLM task and its description. 

{{ hf_ai_tasks }}

Your task is to identify one or more LLM tasks for the context given below. Respond only with a JSON list (maximum length {{ limit }} items) containing the most relevant LLM task labels. Do not include description. Ensure that your answer only includes the json list.

Context: {{ usecase }}
Output: """
