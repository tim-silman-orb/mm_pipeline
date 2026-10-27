# MM Pipeline

`mm_pipeline` is a standalone, configuration-driven media-monitoring data
collection pipeline. It is intended to run in a local shell or as an ECS task.
Project-specific topics and source selections live in YAML; collection,
normalisation and storage behaviour live in reusable Python code.

The folder has no dependency on the parent `MM_Ebola_Tracker` repository and
can be moved alongside it.

## Data flow

```text
CLI -> validated project config -> registered source adapters
    -> topic filter -> deduplication -> validation
    -> optional Bedrock extraction -> local or S3 output + run manifest
```

Every run uses a half-open UTC window: `start <= published_at < end`. The
window is included in the deterministic run ID, so replaying the same window
writes to the same destination.

## Installation

```powershell
cd mm_pipeline
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,telegram]"
```

## Configuration

- `config/source_catalog.yml` contains reusable feed and website definitions.
- `config/projects/ebola.yml` contains Ebola-specific selection and terms.
- `config/projects/example.yml` demonstrates a second topic using the same code.
- Secret values are never stored in YAML. Configuration names the environment
  variables from which adapters should read them.

For the Ebola configuration, set at least:

```text
OPOINT_TOKEN=...
```

Telegram and X variables are only required when those disabled sources are
enabled. A local `.env` can be passed explicitly with `--env-file`. On ECS,
inject secrets through the task definition and use the task role for AWS.

## Commands

Validate configuration without calling external services:

```powershell
monitor-pipeline validate-config --config config/projects/ebola.yml
```

List available adapters:

```powershell
monitor-pipeline list-adapters
```

Run an explicit, reproducible collection window:

```powershell
monitor-pipeline run `
  --config config/projects/ebola.yml `
  --start 2026-10-20T00:00:00Z `
  --end 2026-10-27T00:00:00Z `
  --env-file .env
```

Run selected sources only:

```powershell
monitor-pipeline run `
  --config config/projects/ebola.yml `
  --lookback-days 1 `
  --source cdc_newsroom `
  --source who_disease_outbreak_news
```

Override local output with S3:

```powershell
monitor-pipeline run `
  --config config/projects/ebola.yml `
  --lookback-days 7 `
  --output-uri s3://my-monitoring-bucket/runs
```

## Outputs

Each run writes:

```text
<base>/<project>/<project>_<start>_<end>/
├── master_data.jsonl
├── developments.jsonl       # only when extraction is enabled
└── manifest.json
```

The manifest records source counts, warnings, failures, filtering counts,
configuration hash and final status. Required-source or empty-dataset failures
produce a non-zero CLI exit code after the manifest is written.

## Source adapters

The built-in adapter registry supports:

- `rss`: configuration-driven RSS/Atom feeds.
- `html_list`: a page containing links selected with CSS.
- `html_page`: a single monitored webpage.
- `opoint`: an OPoint stored search.
- `telegram`: configured Telegram channels.
- `x`: an X recent-search query.

An adapter accepts a source configuration and UTC collection window and returns
canonical `Article` records. A new mechanism is added by implementing the same
interface and registering it in `registry.py`; a new RSS feed only requires a
catalog entry.

## Validation notebook

`validation/pipeline_checks.ipynb` contains executable assertions for
configuration loading, filtering, deduplication and a complete pipeline run
using a fake source. It does not call external services or require AWS
credentials.

Open it in JupyterLab and run all cells, or execute it non-interactively:

```powershell
jupyter nbconvert `
  --execute `
  --to notebook `
  --output-dir output `
  validation/pipeline_checks.ipynb
```

The pipeline accepts adapter, storage and extractor dependencies so the
notebook can validate orchestration without network access. Unlike a pytest
suite, the notebook is not checked automatically unless this command is added
to CI.

### Live comparison with the legacy Ebola pipeline

`validation/pre_bedrock_parity.ipynb` is a separate, live integration check.
It runs the legacy Ebola collection path and this package for approximately
the same UTC lookback window, disables extraction, and compares the two master
datasets before anything is sent to Bedrock.

Install the development dependencies, make the legacy project available next
to this folder, and put the OPoint token in either project's `.env` as `TOKEN`
or `OPOINT_TOKEN`:

```powershell
pip install -e ".[dev]"
jupyter lab validation/pre_bedrock_parity.ipynb
```

Run all cells. The notebook calls the live OPoint and public-health sources,
so it requires network access and may take several minutes. It does not call
Bedrock. Reports and copies of both pre-extraction datasets are written below
`parity_output/<run timestamp>/`, which is ignored by Git.

Because the two live collections are sequential and the legacy OPoint helper
chooses its end time when called, differences are evidence to investigate—not
automatically regressions. The report separates missing IDs, additional IDs,
field differences, source counts, and source failures. A later repeatable
regression suite can replay captured responses if exact deterministic equality
is required.

## ECS execution

The included Dockerfile starts with the `monitor-pipeline` command. An ECS
container command override can therefore be:

```json
[
  "run",
  "--config", "config/projects/ebola.yml",
  "--start", "2026-10-20T00:00:00Z",
  "--end", "2026-10-27T00:00:00Z",
  "--output-uri", "s3://my-monitoring-bucket/runs"
]
```

Do not set `AWS_PROFILE` in ECS. boto3 will use the task role. Grant that role
only the required S3, Bedrock and secret permissions.
