#!/bin/sh

docker exec $((basename $PWD) | tr '[A-Z]' '[a-z]')-jupyterlab-1 python -m pip install -r ./code/requirements.txt
