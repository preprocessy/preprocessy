""" setup """
import io

from setuptools import find_packages
from setuptools import setup

with io.open("README.md", "rt", encoding="utf8") as f:
    LONG_DESC = f.read()

VERSION = "1.0.3"

# This call to setup() does all the work
setup(
    name="preprocessy",
    version=VERSION,
    description="Data Preprocessing framework that provides customizable pipelines.",
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
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "alive-progress~=1.6.2",
        "colorama>=0.4.4",
        "pandas>=1.0.5",
        "prettytable>=2.1.0",
        "scikit-learn>=0.23.2",
        "stringcase>=1.2.0"
        # add contents from requirements.txt only
        # "sample_package>=version_number"
    ],
)
