from typing import List
from jinja2 import Template


class ZeroShotPromptBuilder:

    def __init__(self, prompt_data, prompt_template):
        self.prompt_data = prompt_data
        self.prompt_template = prompt_template

    def build(self, **kwargs) -> List[str]:
        """Build prompts using the data samples

        Returns:
            Prompts List[str]: A List of prompt.
        """

        if isinstance(self.prompt_data, dict):
            return Template(self.prompt_template).render(
                data=self.prompt_data, **kwargs
            )
        elif isinstance(self.prompt_data, list):
            prompts = []
            for data in self.prompt_data:
                prompts.append(
                    Template(self.prompt_template).render(data=data, **kwargs)
                )

            return prompts


class FewShotPromptBuilder:

    def __init__(self, prompt_data, prompt_template):
        self.prompt_data = prompt_data
        self.prompt_template = prompt_template

    def build(self, **kwargs) -> List[str]:
        """Build prompts using the data samples

        Returns:
            Prompts List[str]: A List of prompt.
        """

        if isinstance(self.prompt_data, dict):
            assert (
                "examples" in self.prompt_data
            ), f"When using the Few shot `build` API, `prompt_data` must include examples. Element:\n {self.data} does not have examples."

            return Template(self.prompt_template).render(
                data=self.prompt_data, **kwargs
            )

        elif isinstance(self.prompt_data, list):
            prompts = []
            for data in self.prompt_data:
                assert (
                    "examples" in data
                ), f"When using the Few shot API, `prompt_data` must include examples. Element:\n {data} does not have examples."

                assert "examples" in data and isinstance(
                    data["examples"], list
                ), f"When using the Few shot `build` API, `prompt_data` examples must be a list."

                assert all(
                    len(example) == len(data["examples"][0])
                    for example in data["examples"]
                ), f"All few-shot examples must be of the same length."

                prompts.append(
                    Template(self.prompt_template).render(data=data, **kwargs)
                )

            return prompts
