from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\n')
    return requirements 

setup(
    name='diffractive_optical_model',
    version='0.0.1',
    description='A concise description of your project',
    author='Marshall B. Lindsay',
    author_email='your_email@example.com',
    packages=find_packages(),
    install_requires=read_requirements()
)

