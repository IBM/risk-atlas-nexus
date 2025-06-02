import json
import os
from pathlib import Path


PACKAGEDIR = Path(__file__).parent.absolute()


def get_data_path():
    return os.path.join(PACKAGEDIR, "knowledge_graph")


def load_resource(file_name):
    return json.loads(
        Path(os.path.join(PACKAGEDIR, "templates", file_name)).read_text()
    )
