.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

define DOWNLOAD_PYSCRIPT
import urllib.request, sys

urllib.request.urlretrieve(sys.argv[1],sys.argv[2])
endef
export DOWNLOAD_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
DOWNLOAD := python -c "$$DOWNLOAD_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr test-resources

lint: ## check style with flake8
	flake8 --max-complexity 10 --ignore E203,E501,W503 autocnet tests

test: ## run tests quickly with the default Python
	pytest autocnet

test-resources: ## Download what we need for testing
	mkdir test-resources
	$(DOWNLOAD) https://hirise-pds.lpl.arizona.edu/PDS/EDR/PSP/ORB_010500_010599/PSP_010502_2090/PSP_010502_2090_RED5_0.IMG test-resources/PSP_010502_2090_RED5_0.img
	$(DOWNLOAD) https://hirise-pds.lpl.arizona.edu/PDS/EDR/PSP/ORB_010500_010599/PSP_010502_2090/PSP_010502_2090_RED5_1.IMG test-resources/PSP_010502_2090_RED5_1.img


test-all: ## run tests on every Python version with tox
	tox

coverage: test  ## check code coverage quickly with the default Python
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/autocnet.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ autocnet
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install
