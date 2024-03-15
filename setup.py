from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """Read requirements from a file and skip the editable install line."""
    with open(file_path) as file_obj:
        requirements = [line.strip() for line in file_obj.readlines()]
    return [req for req in requirements if req and req != HYPHEN_E_DOT]


setup(
    name="mlops-project",
    version="0.0.1",
    author="sajanpoudel",
    author_email="sajan123poudel4@gmail.com",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages(),
)
