from setuptools import setup, find_packages

# Utility function to read the requirements.txt file
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='diffractive_optical_model',
    version='0.0.1',
    description='A diffractive optical model for simulating planar wavefront propagation.',
    author='Marshall B. Lindsay',
    author_email='mblgh6@umsystem.edu',
    packages=find_packages(),
    install_requires=read_requirements(),  # Read from requirements.txt
    include_package_data=True,  # This ensures non-code files are included
    package_data={
        # Specify any additional files to include in the package, such as data files, YAML configs, etc.
        '': ['config.yaml', 'data/**/*.bmp', 'data/**/*.pgm', 'data/**/*.ppm']
    }
)

