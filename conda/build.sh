#!/bin/bash

$PIP install redis pysis

$PYTHON setup.py install --single-version-externally-managed --record record.txt
