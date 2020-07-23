#!/bin/bash


export LD_LIBRARY_PATH=$(HOME)/lib:/usr/local/lib:/lib:/usr/lib

pip install https://files.pythonhosted.org/packages/a7/7c/24fb0511df653cf1a5d938d8f5d19802a88cef255706fdda242ff97e91b7/redis-3.5.3-py2.py3-none-any.whl

python setup.py install --single-version-externally-managed --record record.txt
