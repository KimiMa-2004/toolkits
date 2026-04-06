# TaskLog - Daily Task Execution Tracker

A lightweight Python helper for managing **daily task execution status** and preventing duplicate runs. It keeps a JSON record of which tasks have run successfully today, automatically **skips** duplicates, and **retries** failed runs.

## ✨ Features

- ✅ **Automatic Skip**: Skip tasks that have already been successfully executed today
- 🔄 **Auto Retry**: Automatically retry tasks that failed today
- 📊 **Status Tracking**: Track execution status, timestamps, and error messages
- 🎯 **Flexible Usage**: Use as decorator or function wrapper
- 💾 **Persistent Storage**: JSON-based record storage
- 🧹 **Easy Management**: Built-in functions to view, reset, and clean records
- 🚀 **Zero Dependencies**: Uses only Python standard library

## 📦 Installation / Layout

This module is self‑contained. You can simply copy the `tasklog` folder to your project:

```bash
your_project/
├── tasklog/
│   ├── __init__.py
│   ├── tracker.py
│   ├── utils.py
│   └── record.json  # Auto-generated
└── your_script.py
```

Then import directly:

```python
from tasklog import daily_run, operate, show_records
```

If you install this repo as a package named `toolkit`, you can also use:

```python
from toolkit.tasklog import daily_run, operate, show_records
```

## 🚀 Quick start

### 1. Use as a decorator (recommended)

```python
from tasklog import daily_run

@daily_run()  # default: skip if already succeeded today
def sync_data():
    print("Pulling data from remote API...")
    # ... do work ...
    return "ok"

if __name__ == "__main__":
    sync_data()  # First call runs; repeated calls the same day are skipped
```

### 2. Use as a function wrapper

```python
from tasklog import operate

def generate_report():
    print("Generating report...")
    return "report.pdf"

# Will run once per day (unless failed or forced)
result = operate(generate_report, name="daily_report")
```

### 3. Integrate with your own logger

You can pass a `logging.Logger` instance (for example from `toolkit.logger.get_logger`):

```python
from tasklog import daily_run
from toolkit.logger import get_logger

logger = get_logger(name="tasks", filename="tasks")

@daily_run(logger=logger)
def training_job():
    # ...
    return "done"
```

All task status messages will be logged instead of printed to stdout.

## 🧠 Core APIs

All public functions are exported from `tasklog.__init__`:

- **Execution control**
  - `daily_run(name=None, force=False, logger=None)` – decorator that manages daily runs
  - `daily_run_with_params(name=None, force=False, cache_params=True, logger=None)` – decorator that **includes parameters** in the task key (useful when the same function is called with different args)
  - `operate(func, name=None, force=False, logger=None, **kwargs)` – one‑off wrapper that records execution result

- **Status query**
  - `is_updated_today(name)` – check if a task has succeeded today
  - `get_record(name)` – get raw record dict for a task
  - `get_failed_tasks(today_only=True)` – list of failed task names

- **Display**
  - `show_records(show_failed_only=False, logger=None)` – pretty print (or log) the current records table

- **Record management**
  - `update_record(name, status='success', extra_info=None)` – low-level manual update
  - `reset_record(name, logger=None)` – remove one task record
  - `reset_all_records(logger=None)` – clear all records
  - `reset_failed_records(logger=None)` – clear only failed records
  - `clear_old_records(days=30, logger=None)` – drop records older than `N` days

- **Utilities**
  - `load_records()` / `save_records(records)` – load/save the underlying JSON
  - `ensure_record_file()` – create the record file if missing
  - `set_record_file(path)` – change where the JSON file is stored

## 🗂 Record file

By default, TaskLog uses a JSON file (e.g. `record.json`) to store all execution records. You can control its location via `set_record_file`:

```python
from tasklog import set_record_file

set_record_file("data/tasklog_records.json")
```

Each entry looks roughly like:

```json
{
  "sync_data": {
    "date": "2026-03-13",
    "status": "success",
    "update_time": "2026-03-13 10:15:30",
    "extra_info": {
      "result_type": "str"
    }
  }
}
```

## 📊 Inspecting and cleaning records

```python
from tasklog import show_records, reset_record, clear_old_records, get_failed_tasks

show_records()                    # show all records
show_records(show_failed_only=True)

failed_today = get_failed_tasks(today_only=True)
print("Failed today:", failed_today)

reset_record("sync_data")         # force re-run next time
clear_old_records(days=30)        # keep only the last 30 days
```

This module is intentionally small and dependency‑free, suitable for cron jobs, ETL pipelines, or any script that should only run certain tasks **once per day**.