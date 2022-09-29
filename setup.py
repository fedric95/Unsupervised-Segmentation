import setuptools

MINIMAL_REQUIREMENTS = [
    'torch',
    'pytorch-lightning',
    'numpy',
    'skimage'
]
with open("README.md", encoding="UTF-8") as readme_file:
    readme = readme_file.read()

setuptools.setup(
    name='useg',
    use_scm_version={
        "write_to": "useg/_version.py",
        "write_to_template": '__version__ = "{version}"\n',
    },    
    description='Implementation of some Deep Learning Unsupervised Segmentation models',
    long_description=readme,
    author_email='ricciuti.federico@gmail.com',
    license='Proprietary',
    url='https://github.com/fedric95/Unsupervised-Segmentation',
    packages=setuptools.find_packages(),
    install_requires=MINIMAL_REQUIREMENTS
)
