from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\n')
    return requirements 

setup(
    name='diffractive_optical_model',
    version='0.0.1',
    description='A diffractive optical model for simulating planar wavefront propagation.',
    author='Marshall B. Lindsay',
    author_email='mblgh6@umsystem.edu',
    packages=find_packages(),
)

