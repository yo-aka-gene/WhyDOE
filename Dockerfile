FROM jupyter/datascience-notebook:lab-3.5.0

RUN pip install --upgrade pip
RUN pip install jupyterlab
RUN jupyter serverextension enable --py jupyterlab

WORKDIR $HOME
RUN mkdir doe_modules jupyternb out

EXPOSE 8888
VOLUME ["/home/jovyan/code"]

CMD ['/bin/bash']
