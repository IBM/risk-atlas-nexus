QUESTIONNAIRE_COT_TEMPLATE = """
        I want you to play the role of a compliance officer and answer the question based on the given Intent.
        Return the question, answer and explanation in a json format where question, answer and explanation are keys of the json exactly as shown in the examples.
        you should answer the question followed by an explanation on how that answer was generated.
{% if cot_examples is not none %}{% for example in cot_examples %}
        Intent: {{ example.intent }}
        Question: {{ question }}
        Answer: {{ example.answer }}
{% if example.explanation is not none %}        Explanation: {{ example.explanation }}{% endif %}
{% endfor %}{% endif %}
        Intent: {{ usecase }}
        Question: {{ question }}
"""

RISK_IDENTIFICATION_TEMPLATE = """This is an AI risk similarity task. You are given JSON titled RISKS which contains a list of risk categories and their descriptions.

RISKS:
{{ risks }}

Instructions:
1. Identify the RISKS which are semantically similar and most relevant to the Input context
2. For the similar risks, choose which relation characterizes the relevance relationship between Input context and the RISK.
3. If input doesn't fit into any of the above categories, classify it as Unknown.
4. Respond with a (maximum length 5 items) list of attribute 'category' containing the most relevant classification labels and their relation.

{% if cot_examples is not none %}EXAMPLES:{% for example in cot_examples %}
Input: {{ example.Input }}
Output: {{ example.Output }}{% endfor %}{% endif %}
===== END OF EXAMPLES ======

Input: {{ usecase }}
Output: """

AI_TASKS_TEMPLATE = """Study and understand the JSON below containing a list of LLM task and its description.

{{ hf_ai_tasks }}

Your task is to identify one or more LLM tasks for the context given below. Respond only with a JSON list (maximum length {{ limit }} items) containing the most relevant LLM task labels. Do not include description. Ensure that your answer only includes the json list.

Context: {{ usecase }}
Output: """
