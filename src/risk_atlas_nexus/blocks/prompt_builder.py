from typing import List
from jinja2 import Template


class ZeroShotPromptBuilder:

    def __init__(self, data, prompt_template):
        self.data = data
        self.prompt_template = prompt_template

    def build_one(self, sample_index: int, **kwargs) -> List[str]:
        """Get a single prompt based on the `sample_index`

        Args:
            sample_index (int): index of the data sample to be used in the prompt

        Returns:
            Prompt str: A prompt.
        """
        # Get the desired sample
        data = self.data[sample_index]

        return Template(self.prompt_template).render(data=data, **kwargs)

    def build_all(self, **kwargs) -> List[str]:
        """Build all samples into prompts

        Returns:
            Prompts List[str]: A List of prompt.
        """
        prompts = []
        for data in self.data:
            prompts.append(Template(self.prompt_template).render(data=data, **kwargs))

        return prompts


class FewShotPromptBuilder:

    def __init__(self, cot_data, prompt_template):
        self.cot_data = cot_data
        self.prompt_template = prompt_template

    def build_one(self, sample_index: int, **kwargs) -> List[str]:
        """Get a single prompt based on the `sample_index`

        Args:
            sample_index (int): index of the data sample to be used in the prompt

        Returns:
            Prompt str: A prompt.
        """

        # Get the desired sample
        data = self.cot_data[sample_index]

        assert (
            "examples" in data
        ), f"When using the Few shot `build_one` API, `cot_data` must include examples. Element:\n {data} does not have examples."

        return Template(self.prompt_template).render(data=data, **kwargs)

    def build_all(self, **kwargs) -> List[str]:
        """Build all samples into prompts

        Returns:
            Prompts List[str]: A List of prompt.
        """
        prompts = []
        for data in self.cot_data:
            assert (
                "examples" in data
            ), f"When using the Few shot API, `cot_data` must include examples. Element:\n {data} does not have examples."

            assert "examples" in data and isinstance(
                data["examples"], list
            ), f"When using the Few shot `build_many` API, `cot_data` examples must be a list."

            assert all(
                len(example) == len(data["examples"][0]) for example in data["examples"]
            ), f"All few-shot examples must be of the same length."

            prompts.append(Template(self.prompt_template).render(data=data, **kwargs))

        return prompts
