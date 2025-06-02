from typing import Union

from linkml_runtime.dumpers.dumper_root import Dumper
from linkml_runtime.utils.context_utils import CONTEXTS_PARAM_TYPE
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel


class LatexDumper(Dumper):

    def dump(
        self,
        element: Union[BaseModel, YAMLRoot],
        to_file: str,
        contexts: CONTEXTS_PARAM_TYPE = None,
        **kwargs,
    ) -> None:
        """
        Write element as latex to to_file

        Args:
            element: Union[BaseModel, YAMLRoot]
                LinkML object to be output
            to_file: str
                file to write to
            contexts: Optional[Union[CONTEXT_TYPE, List[CONTEXT_TYPE]]]
                 a list of JSON-LD contexts, which can be one of:
                    * the name of a JSON-LD file
                    * the URI of a JSON-lD file
                    * JSON-LD text
                    * A JsonObj object that contains JSON-LD
                    * A dictionary that contains JSON-LD
        """
        if isinstance(element, BaseModel):
            element = element.dict()

        super().dump(element, to_file, contexts=contexts, **kwargs)

    def dumps(self, element: Union[BaseModel, YAMLRoot], **_) -> str:
        """
        Return element as latex string

        Args:
            element: Union[BaseModel, YAMLRoot],
                LinkML object to be emitted
            _:
                method specific arguments

        Returns:
            str
        """
        if isinstance(element, BaseModel):
            element = element.dict()
            element = self._to_tex_from_dict(element)
        return element

    def _to_tex_from_dict(self, element_dict) -> str:
        """
        Format as tex

        Args:
            element_dict: dict

        Returns:
            str
        """
        tex = [
            "\\documentclass[a4paper,12pt]{article}\n",
            "\\usepackage{xcolor}\n",
            "\\usepackage[skins]{tcolorbox}%\n",
            "\\usepackage{tcolorbox}\n",
            "\\usepackage{hyperref}\n",
            "\\newtcolorbox[auto counter, number within=section]{definitionbox}[2][]{%\n",
            "colframe=black!50,\n",
            "colback=black!5,\n",
            "coltitle=white,\n",
            "fonttitle=\\bfseries,\n",
            "title={#2},\n",
            "sharp corners=south,\n",
            "enhanced,\n",
            "before upper={\\noindent},\n",
            "}\n",
            "\\begin{document}\n",
            "\\section{IBM AI Risk Atlas}\n",
            "Explore this atlas to understand some of the risks of working with generative AI, foundation models, "
            "and machine learning models. Risks are categorized with one of these tags:\n",
            "\\begin{itemize}\n",
            "\\item Traditional AI risks (applies to traditional models as well as generative AI)\n",
            "\\item Risks amplified by generative AI (might also apply to traditional models)\n",
            "\\item New risks specifically associated with generative AI\n",
            "\\end{itemize}\n",
            "\\subsection*{AI Risk Atlas Definitions}\n",
        ]

        for risk in element_dict["risks"]:
            tex.append("\\begin{definitionbox}{" + risk["name"] + "}\n")
            tex.append(
                risk["description"].replace("’", "'").replace(" ", " ")
                + "\\newline\\newline\n"
            )
            tex.append(
                "\\textbf{Concern: }"
                + risk["concern"].replace("’", "'").replace(" ", " ")
                + "\\newline\\newline\n"
            )
            tex.append(
                "\\textbf{Type: }"
                + risk["type"].replace("’", "'").replace(" ", " ")
                + "\\newline\n"
            )
            tex.append(
                "\\textbf{Descriptor: }" + risk["descriptor"] + " \\newline\\newline\n"
            )
            tex.append("\\textbf{Implementation details: }" + " \\newline\n")
            tex.append("ID: " + risk["id"] + " \\newline\n")
            tex.append("Tag: " + risk["tag"] + " \\newline\n")
            tex.append(
                "URI:  \\href{"
                + risk["url"]
                + "}{IBM AI Risk Atlas - "
                + risk["name"]
                + "}\\newline\n"
            )
            tex.append("\\end{definitionbox}\n")
        tex.append("\\end{document}\n")
        tex_all = "".join(tex)
        return tex_all
