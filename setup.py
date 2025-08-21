from setuptools import setup, find_packages

setup(
    name="causalllm",
    version="0.1.0",
    description="Causal reasoning framework for language models",
    author="Durai Rajamanickam",
    packages=find_packages(include=["causalllm", "causalllm.*"]),
    install_requires=[
        "networkx",
        "matplotlib",
        "openai",
        "langchain",
        "pyyaml",
    ],
    python_requires=">=3.7",
)
