We welcome contributions from developers of all levels. If you find anything amiss with the notebooks or dependencies, please feel free to create a pull request.

If you want to change the conda dependency specification (the yaml file), you can test it like this:
```bash
conda env create --file langchain_ai.yaml --force
```

You can update the pip requirements like this:
```bash
pip freeze > requirements.txt
```

Please make sure that you keep these two ways of maintaining dependencies in sync.

Then make sure, you test the notebooks in the new environment to see that they run.

# Code validation
We've included a `Makefile` that includes instructions for validation with flake8, mypy, and other tools. I have run mypy like this:
```bash
make typecheck
```

To run the code validation in ruff, please run
```bash
ruff check .
```
    
