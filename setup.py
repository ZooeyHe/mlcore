from setuptools import setup, find_packages

setup(
    name="mlcore",
    version="1.0",
    description="Python Machine Learning Core",
    author="Zooey He",
    author_email="zhuohonghe@gmail.com",
    packages=["mlcore"],
    requires=[
        "yacs"
    ]
)