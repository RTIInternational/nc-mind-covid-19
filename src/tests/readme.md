# Tests

## Execution

To run the tests:

(from the root directory, not this one)

```
python -m pytest --pyargs src --cov=src --cov-config=.coveragerc
```

or

```
docker-compose run --rm covid bash -c "python -m pytest --pyargs src --cov=src --cov-config=.coveragerc"
```

If you're doing both options (docker and local) for running tests, you might encounter an error
like this: `HINT: remove __pycache__ / .pyc files and/or use a unique basename for your test file model run`. If you do, run:

```
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf`
```
from the root directory. This is occurring because when the tests are run in one environment, the absolute paths to the cached python files are in different locations.

### Slow Model Output Tests

The model output tests are meant to evaluate the raw and aggregated output files from running multiple iterations and multiple runs of the model. These tests require that the model be first run 4 times, which takes about 3-5 minutes.

These tests are skipped by default, and can be enabled by calling pytest with the `--runslow` flag.
