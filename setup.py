from re import M
import setuptools

MINIMAL_REQUIREMENTS = [
    'torch>=1.11.0',
    'torchvision>=0.12.0',
    'pytorch-lightning>=1.5.9',
    'numpy>=1.19.5',
    'skimage>=0.18.3',
    'sklearn>=1.0.2',
    'tqdm>=4.64.0',
    'neptune-client>=0.14.3'
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
