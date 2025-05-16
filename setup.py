from setuptools import setup, find_packages

setup(
    name="effibench",
    version="0.1.0",
    description="Efficiency Benchmarking for LLM Code Generation",
    author="EffiBench Team",
    author_email="qyhh@connect.hku.hk",
    packages=find_packages(),
    include_package_data=True,
)