#!/bin/sh

docker exec -it $((basename $PWD) | tr '[A-Z]' '[a-z]')-jupyterlab-1 sudo pip list --format=freeze > ./doe_modules/requirements.txt
