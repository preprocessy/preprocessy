Firstly, thank you for taking time and contributing to the project. All your efforts to contribute are highly appreciated!

## Feature Requests

Feature Requests by the community are highly encouraged. However, please make sure to check that your feature is not already on the [roadmap](https://github.com/preprocessy/preprocessy/projects/1). Clearly state why this feature is necessary or why it will be a great addition to the project along with sample use cases to help developers better understand it.

## Reporting an issue

Before submitting an issue please make sure:

- You have read the documentation about the pipeline or function you're trying to use.
- You have already searched for related [issues](https://github.com/preprocessy/preprocessy/issues), and found none open (if you found a related closed issue, please link to it from your post).
- Your issue title is concise, on-topic and polite.
- You can and do provide steps to reproduce your issue.
- For developers, please make sure your environment is setup correctly.

## Workflow for submitting a PR

1. Fork the repository to your own GitHub account

2. Clone from your repository

```bash
    $ git clone https://github.com/your-username/repository-name
```

3. Create a virtual environment (always a good practice!) and install the development dependencies

```bash
    $ pip install -r requirements_dev.txt
```

4. Make the necessary changes

5. Run the tests

```bash
    $ pytest -v -s
```

6. Format the code before committing (replace `./venv` with your virtual environment folder)

```bash
    $ black . && codespell --skip=".git,*.gif,*.png,*.PNG,./venv,*.json,./datasets,.DS_Store,*.pyc,./htmlcov,.coverage"
```

7. Create a new pull request with an appropriate title, detailed explanation of what the pull request does and attach links to other issues or pull requests related to your pull request

## Running test coverage

Generating a report of lines that do not have test coverage can indicate where to start contributing. Run `pytest` using `coverage` and generate a report.

```bash
    $ coverage run --source=./preprocessy -m pytest

    $ coverage html
```

Open `htmlcov/index.html` in your browser to explore the report.

Read more about [coverage](https://coverage.readthedocs.io/en/coverage-5.4/).

## License

By contributing your code to the Preprocessy GitHub repository, you agree to license your contribution under the [MIT](https://github.com/preprocessy/preprocessy/blob/master/LICENSE) license.
