# Unit-tests for ML Pipelines in Python (sklearn)

### Environment

#### Using [Virtualenv](https://virtualenv.pypa.io/en/latest/)

```
[sudo] pip install virtualenv
virtualenv [-p python3] <env_name>
echo <env_name> >> .gitignore
<env_name>/bin/activate
pip install -r requirements.txt

# To deactivate your environment
deactivate

```

#### Using [Miniconda](https://conda.io/miniconda.html)

```
conda create -n <env_name> [python=<python_version>]
source activate <env_name>
pip install -r requirements.txt

# To deactivate your environment
source deactivate
```

### Project Structure

```
├── resources/
│   ├── data/                 <- Input data folder
│   └── results/              <- Results folder
├── tfdv/                     <- Scripts to compare the system against
|                                TFX and data-linter
├── third_party/              <- Data-linter and facets source code
├── analyzers.py              <- DataFrameAnalyzer
├── error_generation.py       <- Error generation utilities
├── evaluation.py             <- Evaluation utilities, tests
├── hilda.py                  <- HILDA'19 showcase
├── messages.py               <- Text messages placeholder
├── models.py                 <- ML models
├── openml.py                 <- Utilities for using OpenML
├── pipelines.py              <- Pipelines
├── profilers.py              <- DataFrameProfiler, PipelineProfiler
├── selection.py              <- RandomSelector, PairSelector
├── settings.py               <- Helper functionality
├── shift_detection.py        <- Dataset shift detection utilities
├── test_suite.py             <- TestSuite, AutomatedTestSuite
├── transformers.py           <- Custom transformers for sklearn pipeline
└── visualization_utils.py    <- Visualization utilities
```

### Entry Points

```
hilda.py                      <- Showcase on automated unit tests functionality
evaluation.py                 <- Checks whether errors in data crash the
                                 serving system or affect performance of the
                                 pipelines, and whether unit tests detect these
                                 errors
shift_detection.py            <- Snowcase on dataset shift detection
```
