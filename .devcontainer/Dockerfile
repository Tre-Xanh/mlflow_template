# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.238.0/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

RUN conda install mamba -c conda-forge
COPY environment.yml /tmp/conda-tmp/
RUN mamba env update -n dev -f /tmp/conda-tmp/environment.yml

ENV MLFLOW_CONDA_CREATE_ENV_CMD="mamba"
