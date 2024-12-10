from setuptools import setup, find_packages

setup(
    name="gar-rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'nltk>=3.8.1',
        'spacy>=3.7.2',
    ],
)