#!/bin/bash


export LD_LIBRARY_PATH=/usr/local/lib:/lib:/usr/lib

pip install . --no-deps -vv --prefix=$PREFIX
