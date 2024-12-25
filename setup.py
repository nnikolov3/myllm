from setuptools import setup, find_packages

setup(
    name="myllm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'langchain_core',
        'langchain_ollama',
        'langchain_chroma',
        'rich',
        'torch',
        'psutil',
        'numpy',
        'scikit-learn'
    ],
)