Contributing
============

Firstly, thank you for taking time and contributing to the project. All your
efforts to contribute are highly appreciated!

Support Questions
-----------------

Please don't use the issue tracker for this. The issue tracker is a tool to
address bugs and feature requests in Preprocessy itself. Use one of the following
resources for questions about using Preprocessy or issues with your own code:

- The ``#questions`` channel on our Discord Chat: https://discord.gg/5q2yCqqU6N

Feature Requests
----------------

Feature Requests by the community are highly encouraged. However, please make
sure to check that your feature is not already on the `roadmap`_. Clearly state
why this feature is necessary or why it will be a great addition to the project
along with sample use cases to help developers better understand it.

.. _roadmap: https://github.com/preprocessy/preprocessy/projects/1

Reporting an issue
------------------

Before submitting an issue please make sure:

- You have read the documentation about the pipeline or function you're trying to use.
- You have already searched for related `issues`_, and found none open (if you found a related closed issue, please link to it from your post).
- Your issue title is concise, on-topic and polite.
- You can and do provide steps to reproduce your issue.
- For developers, please make sure your environment is setup correctly.
- List your Python and Preprocessy versions. Also mention if the issue occurred while using Jupyter Notebook or a Python Script. If possible, check if this issue is already fixed in the latest releases or the latest code in the repository.

.. _issues: https://github.com/preprocessy/preprocessy/issues

Submitting patches
------------------

If there is not an open issue for what you want to submit, prefer
opening one for discussion before working on a PR. You can work on any
issue that doesn't have an open PR linked to it or a maintainer assigned
to it. These show up in the sidebar. No need to ask if you can work on
an issue that interests you.

Include the following in your patch:

-   Use `Black`_ to format your code. This and other tools will run
    automatically if you install `pre-commit`_ using the instructions
    below.
-   Include tests if your patch adds or changes code. Make sure the test
    fails without your patch.
-   Update any relevant docs pages and docstrings.
-   Add an entry in ``CHANGES.rst``. Use the same style as other
    entries. Also include ``.. versionchanged::`` inline changelogs in
    relevant docstrings.

.. _Black: https://black.readthedocs.io
.. _pre-commit: https://pre-commit.com

Workflow for submitting a PR
----------------------------

1. Fork the repository to your own GitHub account

2. Clone from your repository

.. code-block:: text

    $ git clone https://github.com/your-username/repository-name


3. Create a virtual environment (always a good practice!)

4. Install the development dependencies

Using Poetry

You can use `poetry`_ to install the dependencies of the projects

.. code-block:: text

    $ poetry install

Using Pip

.. code-block:: text

    $ pip install -r requirements_dev.txt


5. Install pre-commit hooks

.. code-block:: text

    $ pre-commit install

6. Make the necessary changes

7. Run the tests

.. code-block:: text

    $ pytest -W ignore::DeprecationWarning


8. Create a new pull request with an appropriate title, detailed explanation of
what the pull request does and attach links to other issues or pull requests
related to your pull request

.. _poetry: https://python-poetry.org

Running test coverage
---------------------

Generating a report of lines that do not have test coverage can indicate where
to start contributing. Run ``pytest`` using ``coverage`` and generate a report.

.. code-block:: text

    $ coverage run --source=./preprocessy --omit="./*/__init__.py"  -m pytest
    $ coverage html

Open ``htmlcov/index.html`` in your browser to explore the report.

Read more about `coverage <https://coverage.readthedocs.io>`__.

Building the docs
-----------------

Build the docs in the ``docs`` directory using Sphinx.

.. code-block:: text

    $ cd docs
    $ make html

Open ``_build/html/index.html`` in your browser to view the docs.

Read more about `Sphinx <https://www.sphinx-doc.org/en/stable/>`__.
