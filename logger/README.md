<!--
 * @Author: Qimin Ma
 * @Date: 2026-03-13 20:49:12
 * @LastEditTime: 2026-03-13 20:49:17
 * @FilePath: /Toolkit/logger/README.md
 * @Description: 
 * Copyright (c) 2026 by Qimin Ma, All Rights Reserved.
-->

# Logger (`toolkit.logger`)

A tiny helper around Python's built-in `logging` to quickly get a logger that writes to **console**, **file**, or both, with sensible defaults and Windows‑friendly log file deletion.

If you vendor this folder directly into your project, you can either:

- `from logger import get_logger, delete_logger_file` (if `logger/` is on `PYTHONPATH`)
- `from toolkit.logger import get_logger, delete_logger_file` (if you install this repo as the `toolkit` package)

## Environment variables

- **`LOG_LEVEL`**: Default log level (e.g. `DEBUG`, `INFO`, `WARNING`). Defaults to `INFO`.
- **`LOGGER_DIR`**: Directory where log files are created. Defaults to `./logs`.

`.env` is loaded via `python-dotenv`, so you can configure these in a local `.env` file:

```env
LOG_LEVEL=DEBUG
LOGGER_DIR=./logs
```

## `get_logger(...)`

Return a configured `logging.Logger` instance. You can use it with console only, file only, or both.

```python
from logger import get_logger  # or: from toolkit.logger import get_logger

# Console only, default name and level from LOG_LEVEL
logger = get_logger()
logger.info("Application started")
logger.warning("Something to check")

# Also write to file ./logs/my_app.log (or {LOGGER_DIR}/my_app.log)
logger = get_logger(name="my_app", filename="my_app", ifconsole=True)
logger.info("This goes to console and to ./logs/my_app.log")

# File only, no console; custom level
logger = get_logger(name="batch", filename="batch", ifconsole=False, level="DEBUG")
logger.debug("Debug message only in file")
```

**Signature**

```python
get_logger(
    name: str = "smart_dataloader",
    filename: str | None = None,
    level: str | None = None,
    ifconsole: bool = True,
) -> logging.Logger
```

**Parameters**

| Parameter    | Description |
|-------------|-------------|
| `name`      | Logger name (default `"smart_dataloader"`; used with `logging.getLogger(name)`). |
| `filename`  | If set, log to `{LOGGER_DIR}/{filename}.log`. `None` = no file output. |
| `level`     | Log level, e.g. `"DEBUG"`, `"INFO"`. Uses env `LOG_LEVEL` if `None`. |
| `ifconsole` | If `True`, add a console handler (default `True`). |

Handlers for a given logger name are added only once, so repeated `get_logger(...)` calls will not duplicate log lines.

## `delete_logger_file(...)`

Remove a log file by name and close its file handler first, so it works correctly on Windows where open file handles prevent deletion.

```python
from logger import get_logger, delete_logger_file  # or: from toolkit.logger import ...

logger = get_logger(filename="temp_job")
logger.info("Done")

# When you no longer need the file (e.g. after tests):
delete_logger_file("temp_job")  # Deletes ./logs/temp_job.log (or {LOGGER_DIR}/temp_job.log)
```

**Signature**

```python
delete_logger_file(filename: str = "smart_dataloader") -> None
```

`filename` is the same name you passed to `get_logger(..., filename="...")`.

---