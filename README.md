<!--
 * Toolkit - Small Reusable Utilities
 * Modules: timer, logger, tasklog
-->

## Toolkit – Python Utility Collection

**Toolkit** is a small collection of self‑contained Python utilities that you can vendor into any project or install as a package.  
Currently it contains three focused modules:

- **`timer`**: Lightweight timing helper for benchmarking code blocks.
- **`logger`**: Opinionated wrapper around `logging` with console/file output and `.env` support.
- **`tasklog`**: Daily task execution tracker that prevents duplicate runs and records status in JSON.

Each submodule has its own `README.md` with detailed usage and examples.

### Modules

- **`timer`** (`toolkit.timer`)
  - Context‑manager and manual timer with units `ns`, `us`, `ms`, `s`.
  - Great for quick performance checks.

- **`logger`** (`toolkit.logger`)
  - `get_logger(...)` to log to console and/or file.
  - Configurable via environment variables `LOG_LEVEL` and `LOGGER_DIR` (supports `.env` through `python-dotenv`).

- **`tasklog`** (`toolkit.tasklog`)
  - Decorators and helpers to ensure a task runs at most once per day.
  - Tracks success/failure, timestamps, and optional error info in a JSON file.

### Installation

You can either:

- **Vendor** the folders directly into your project (no extra tooling), or
- Turn this repo into a package (e.g. `pip install -e .`) and import via `toolkit.*`.

Minimal runtime dependency is `python-dotenv` (for loading `.env` in `logger`).  
See `requirements.txt` for the exact version pin.

### Basic usage

```python
from toolkit.timer import Timer
from toolkit.logger import get_logger
from toolkit.tasklog import daily_run

logger = get_logger(name="example", filename="example")

@daily_run(logger=logger)
def daily_job():
    with Timer(unit="ms", message="do_work"):
        # ... your work here ...
        pass

if __name__ == "__main__":
    daily_job()
```

For more details, check:

- `timer/README.md`
- `logger/README.md`
- `tasklog/README.md`

