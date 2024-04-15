#!/bin/sh

get_id=$(id)
nb_id=${get_id[@]:4:3}
docker_name=$((basename $PWD) | tr '[A-Z]' '[a-z]')-jupyterlab-1
sed -i '' -e s/CONTAINER_NAME/${docker_name}/ $(dirname $0)/docker-compose.yml
sed -i '' -e s/YOUR_ID/${nb_id}/ $(dirname $0)/docker-compose.yml
git update-index --assume-unchanged $(dirname $0)/docker-compose.yml
