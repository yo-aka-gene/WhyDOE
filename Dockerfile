FROM jupyter/datascience-notebook:lab-3.5.0

USER root
RUN apt-get update && \
    apt-get install -y \
    libcairo2-dev \
    libxt-dev \
    libgirepository1.0-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

RUN pip install --upgrade pip
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY renv.lock /tmp/
WORKDIR /tmp
RUN R -e "install.packages('renv', repos='http://cran.us.r-project.org'); renv::consent(provided = TRUE); renv::restore(library = .libPaths()[1])"

WORKDIR $HOME
RUN mkdir doe_modules jupyternb out

EXPOSE 8888
VOLUME ["/home/jovyan/code"]
