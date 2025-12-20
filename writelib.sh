#!/bin/sh

docker exec $(basename "$PWD" | tr '[A-Z]' '[a-z]')-jupyterlab-1 pip list --format=freeze > ./doe_modules/requirements.txt
