""" setup """

import io

from setuptools import setup

with io.open("README.md", "rt", encoding="utf8") as f:
    LONG_DESC = f.read()

VERSION = "0.0.1"

# This call to setup() does all the work
setup(
    name="preprocessy",
    version=VERSION,
    description="Data Preprocessing library that provides customizable pipelines.",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    url="https://github.com/Saif807380/dpp",
    author="Saif Kazi",
    author_email="saif1204kazi@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(where="./preprocessy"),
    include_package_data=True,
    install_requires=[
        # add contents from requirements.txt only
        # "sample_package>=version_number"
    ],
)
