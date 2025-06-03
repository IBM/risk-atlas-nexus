"""
The prompt builder translates user intent and Chain of Thought (CoT) samples into
instructions for a language model. This guidance helps the model understand the context
and generate relevant, coherent language-based output.

Prompt Templates take input in the form of prompt data, with each key representing a variable
in the prompt template to be filled in. The prompt template then creates formatted instructions
using this prompt data.

There are two types of prompt builders: the `ZeroShotPromptBuilder`, which accepts only a
dictionary of key/value pairs, and the `FewShotPromptBuilder`, which accepts a list of
dictionaries containing key/value pairs and the CoT examples.
"""

from typing import Any, List

from jinja2 import Template


class ZeroShotPromptBuilder:

    def __init__(self, prompt_template: str):
        """_summary_

        Args:
            prompt_template (str): The string template that creates formatted
                instructions using the prompt data.
        """
        self.prompt_template = prompt_template

    def build(self, **prompt_data) -> str:
        """Build prompt using the prompt_data and prompt_template
        Args:
            prompt_data (Dict[str, Any]): properties with each key
                representing a variable in the prompt template to be filled in.
        Returns:
            Prompt str: A formatted instruction prompt for the language model.
        """

        return Template(self.prompt_template).render(**prompt_data)


class FewShotPromptBuilder:

    def __init__(self, prompt_template):
        """_summary_

        Args:
            prompt_template (str): The string template that creates formatted
                instructions using the prompt data.
        """
        self.prompt_template = prompt_template

    def build(self, cot_examples: List[Any], **prompt_data) -> str:
        """Build prompt using the prompt_data and prompt_template
        Args:
            cot_examples (List[Any]): A List of Chain of Thought (CoT) examples
            prompt_data (Dict[str, Any]): properties with each key
                representing a variable in the prompt template to be filled in.
        Returns:
            Prompt str: A formatted instruction prompt for the language model.
        """

        assert (
            cot_examples is not None
        ), f"When using the Few shot API, `cot_examples` cannot be None."

        assert isinstance(
            cot_examples, list
        ), f"When using the Few shot `build` API, `cot_examples` must be a list."

        assert all(
            len(example) == len(cot_examples[0]) for example in cot_examples
        ), f"All few-shot cot_examples must be of the same length."

        return Template(self.prompt_template).render(
            cot_examples=cot_examples, **prompt_data
        )
