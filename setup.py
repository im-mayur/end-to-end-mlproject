from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(path:str)->List[str]:
    '''
    This function reads the requirements file and returns the list of requirements
    '''
    
    requirements=[]
    with open(path) as file_object:
        requirements = file_object.readlines()
        requirements=[req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    # return requirements        






setup(
    name="mlproject",
    version="0.0.1",
    author="mayur",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)