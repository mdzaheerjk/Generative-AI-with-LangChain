**It is highly recommended that you follow the instructions to make sure you have all dependencies and the API keys set up as expected.**

### Environment
You can install your local environment with conda (recommended) or pip. The environment configurations for conda, pip, and poetry are provided. They all have been tested on MacOS. Please note that if you choose pip as you installation tool, you might need additional installation of system dependencies.

If you have any problems with the environment, please raise an issue, where you show the error you got. If you feel confident, please go ahead and create a pull request.

On Windows, some people have been experiencing difficulties with conda and pip (because of readline and ncurses). If that's the case for you, please have a look at [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) or use the Docker installation. Some people on Winodws reported they [needed](https://stackoverflow.com/questions/73969269/error-could-not-build-wheels-for-hnswlib-which-is-required-to-install-pyprojec/76245995#76245995) to install Visual Cpp Build Tools. In any case, if you have any problems with the environment, please raise an issue, where you show the error you got. If you feel confident that you found an improvement, please go ahead and create a pull request.

### Pandoc

For pip and poetry, make sure you install pandoc in your system. On MacOS use brew:
```bash
brew install pandoc
```

On Ubuntu or Debian linux, use apt:
```bash
sudo apt-get install pandoc
```

On Windows, you can use an [installer](https://github.com/jgm/pandoc/releases/latest).


### A working compiler
You should also ensure that your system has a compiler installed, for example like this (on Linux or WSL):
```bash
sudo apt install gcc g++
```

On MacOs, a Homebrew command like this will work:
```bash
brew install gcc
```

On Windows, please install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### Conda
This is the recommended method for installing dependencies. Please make sure you have [anaconda](https://www.anaconda.com/download) installed.

First create the environment for the book that contains all the dependencies:
```bash
conda env create --file=langchain_ai.yaml
```

The conda environment is called `langchain_ai`. You can activate it as follows:
```bash
conda activate langchain_ai 
```

Please note that on Windows, you have to remove the ncurses and readline dependencies (the two lines) from the `.yaml` file. I needed to install the [VS Build Tools](Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/) (which can take about 20 gigs of hard disk).

### Pip
[Pip](https://pypi.org/project/pip/) is the default dependency management tool in Python. With pip, you should be able to install all the libraries from the requirements file:

```bash
pip install -r requirements.txt
```

If you are working with a slow internet connection, you might see a timeout with pip (this can also happen with conda and pip). As a workaround, you can increase the timeout setting like this:
```bash
export PIP_DEFAULT_TIMEOUT=100
```

### Docker
There's a [docker](https://www.docker.com/) file for the environment as well. It uses the docker environment and starts an ipython notebook. To use it, first build it, and then run it:

```bash
docker build -t langchain_ai .
docker run -it -p 8888:8888 langchain_ai
```

You should be able to find the notebook in your browser at [http://localhost:8888](http://localhost:8888).

### Poetry

Make sure you have [poetry](https://python-poetry.org/) installed. On Linux and MacOS, you should be able to use the requirements file:
```bash
poetry install --no-root
```
This should take the `pyproject.toml` file and install all dependencies.

## Setting API keys
Following best practices regarding safety, I am not committing my credentials to GitHub. You might see `import` statements  mentioning a `config.py` file, which is not included in the repository. This module has a method `set_environment()` that sets all the keys as environment variables like this:

Example config.py:

```python
import os

def set_environment():
     os.environ['OPENAI_API_KEY']='your-api-key-here'
```

Obviously, you'd put your API credentials here. Depending on the integration (Openai, Azure, etc) you need to add the corresponding API keys. The OpenAI API keys are the most often used across all the code. 

You can find more details about API credentials and setup in chapter 3 of the book [Generative AI with LangChain](https://www.amazon.com/Generative-AI-LangChain-language-ChatGPT-ebook/dp/B0CBBL55PQ).


## gcloud CLI
For the [Google Vertex models](https://cloud.google.com/vertex-ai) you need to [download and install the Google gcloud CLI](https://cloud.google.com/sdk/docs/install). If you want to do this in Docker, you can adapt your Dockerfile to add this here:

```bash
# 1. Install basic prerequisites
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      gnupg \
      apt-transport-https \
      ca-certificates

# 2. Download & dearmor Google Cloud apt–signing key
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor \
    > /usr/share/keyrings/cloud.google.gpg

# 3. Add the Cloud SDK repository, referencing the keyring
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
    https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list

# 4. Update apt and install the Cloud SDK
RUN apt-get update \
 && apt-get install -y --no-install-recommends google-cloud-cli

# 5. (Optional) Configure gcloud defaults to disable prompts/metrics
RUN gcloud config set disable_usage_reporting true \
 && gcloud config set metrics/environment disable_prompts true
```
