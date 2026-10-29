# mm_pipeline_2

This folder is a standalone, conservative refactor of the Ebola tracker pipeline. It can be copied into its own repository and built without any files from the original `MM_Ebola_Tracker` repository.

The existing collection, extraction, Bedrock and clustering functions are vendored under `src/mm_pipeline_2/backend/`. The new code adds configuration, source classes, stage orchestration and a command-line interface around that retained logic. This intentionally duplicates the old backend while the two implementations are compared.

## Structure

```text
mm_pipeline_2/
|-- config/
|   |-- ebola.yml               # Project, source, model and artifact choices
|   |-- sources.yml             # Existing public-health source definitions
|   `-- publishers.yml          # Existing publisher metadata
|-- src/mm_pipeline_2/
|   |-- backend/                # Vendored legacy domain functions
|   |   |-- fetchers.py         # Public-health feeds, scraping and enrichment
|   |   |-- fetch_support.py    # HTTP retries, caches and Google News resolution
|   |   |-- parsers.py          # Source-specific feed parsers
|   |   |-- utils.py            # Pipeline-only source models and small helpers
|   |   |-- opoint_functions.py # OPoint collection and standardisation
|   |   |-- x_functions.py      # X collection and standardisation
|   |   |-- telegram_functions.py # Telegram collection and standardisation
|   |   |-- extraction_functions.py # Bedrock extraction and checkpoints
|   |   |-- cluster_functions.py # Embeddings and Stage 2 grouping
|   |   `-- common.py           # Bedrock JSON and safe-save helpers
|   |-- collection.py           # Collector classes and master-data preparation
|   |-- cleanup.py              # Stage 3 canonical-story cleanup
|   |-- config.py               # Validated YAML loading and path resolution
|   |-- stages.py               # Collection, extraction, clustering and full run
|   |-- workspace.py            # Local or S3-backed artifact workspace
|   `-- cli.py                  # ECS-friendly command-line interface
|-- tests/test_orchestration.py # Offline orchestration tests
|-- Dockerfile
`-- pyproject.toml
```

The main orchestration classes are:

- `CollectionStage`, which invokes registered source collector classes and builds the canonical master dataset.
- `ExtractionStage`, which invokes the retained `run_extraction` implementation and its resumable CSV checkpoints.
- `ClusteringStage`, which classifies conversation types, runs retained clustering per stream and optionally runs Stage 3 cleanup.
- `FullPipeline`, which runs all three stages with one Bedrock client and artifact workspace.

OOP is limited to stateful boundaries: collectors, stages, the workspace and cleanup. Parsing, prompts and mathematical helpers remain functions where classes would add little value.

The retained backend is not an unedited copy. Project-wide website, briefing,
email and atlas utilities were removed from the pipeline's `utils.py`.
`fetchers.py` keeps the familiar public-health source functions, while its
HTTP retry/cache and Google News machinery is isolated in `fetch_support.py`.

## Install and validate

Run these commands from this folder after it has been copied to its own location:

```powershell
pip install -e ".[dev]"
mm-pipeline validate-config
pytest
```

No import or configuration path refers to the parent repository.

## Commands

Create `master_data.jsonl` from enabled sources:

```powershell
mm-pipeline collect --lookback-days 7
```

Override sources, dates or topic terms for one run:

```powershell
mm-pipeline collect `
  --start-date 2026-10-23 `
  --end-date 2026-10-29 `
  --source opoint `
  --source health `
  --topic ebola `
  --topic bundibugyo
```

Run stages separately for Step Functions:

```powershell
mm-pipeline collect --lookback-days 7
mm-pipeline extract
mm-pipeline cluster --stream news --stream public_health
```

Or run all stages in one ECS task:

```powershell
mm-pipeline run --lookback-days 7
```

`extract --limit 20` provides a bounded development run and is not intended for scheduled production use.

## Credentials and configuration

Locally, the CLI loads `<project_root>/.env` unless `--env-file` is supplied. On ECS, inject credentials through task secrets and grant the task role access to Bedrock and S3. Leave `AWS_PROFILE` unset in ECS so boto3 uses the task role.

The default Ebola configuration expects `TOKEN` for OPoint. Telegram and X credentials are required only if those sources are enabled. Model IDs can remain in YAML or be overridden through the environment variables named in `config/ebola.yml`.

The YAML is project-level configuration. A future project can copy it, change sources/topics/models, and pass it with `--config`; the Python source should not need project-specific edits.

## Artifacts and AWS orchestration

For local runs, `artifact_uri: created_project_data` writes beneath this repository. For separate ECS tasks, use an S3 prefix:

```yaml
artifact_uri: s3://my-monitoring-bucket/ebola/current
```

Each stage downloads that prefix before running and uploads its results afterward. A Step Functions workflow can therefore invoke:

```text
collect -> extract -> cluster
```

EventBridge Scheduler can start either that state machine or a single ECS task using `run`. Failures return a non-zero container exit code.

## Docker

Use this folder itself as the Docker build context:

```powershell
cd mm_pipeline_2
docker build -t mm-pipeline-2 .
docker run --rm mm-pipeline-2 validate-config
```

An ECS command override can be `['collect', '--lookback-days', '7']`.

## Deliberate compatibility choices

- Health `mode: live` incorporates the old ingestion step; `mode: file` reads an existing `health_items.jsonl` snapshot.
- Existing health fetchers accept dates but do not apply them consistently. That behaviour is retained rather than silently changed.
- The retained OPoint function uses a rolling lookback and cannot make a truly historical end-dated query. Explicit dates configure health collection while OPoint receives their inclusive day count.
- Stage 3 cleanup is code rather than notebook state, while retaining its prompt, cache strategy and conservative validation rules.
