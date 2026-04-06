<!--
 * @Author: Qimin Ma
 * @Date: 2026-02-27 03:48:42
 * @LastEditTime: 2026-03-13 20:52:09
 * @FilePath: /Toolkit/timer/README.md
 * @Description: 
 * Copyright (c) 2026 by Qimin Ma, All Rights Reserved.
-->
# Timer (`smart_dataloader.timer`)

The **Timer** class measures elapsed time with configurable units (`ns`, `us`, `ms`, `s`) and supports context manager usage. Useful for benchmarking (e.g. fundamental ops above).

## `Timer(unit="ms", message=None, precision=2)`

- **`unit`**: One of `"ns"`, `"us"`, `"ms"`, `"s"` for display and for `elapsed()`.
- **`message`**: Optional string printed before the elapsed time (e.g. `"Forward-fill NA (Polars)"` → `"Forward-fill NA (Polars) ... 123.45 ms"`).
- **`precision`**: Decimal places for the printed value.

## Context manager (recommended)

```python
from smart_dataloader.timer import Timer

with Timer(unit="ms", message="Forward-fill NA (Polars)"):
    prep.ffill_na()
# Prints: Forward-fill NA (Polars) ... 1684.99 ms
```

## Manual start / stop

```python
from smart_dataloader.timer import Timer

t = Timer(unit="s", message="Total pipeline", precision=3)
t.start()
# ... do work ...
t.stop()
# Prints: Total pipeline ... 12.345 s
print(t.elapsed())       # in seconds
print(t.elapsed("ms"))   # override unit for this call
```

## `TimeUnit` and `TIME_UNIT_SCALE`

- **`TimeUnit`**: Type alias `Literal["ns", "us", "ms", "s"]`.
- **`TIME_UNIT_SCALE`**: Dict mapping unit to factor relative to seconds (e.g. `"ms"` → 1e3).

```python
from smart_dataloader.timer import Timer, TimeUnit, TIME_UNIT_SCALE

# TIME_UNIT_SCALE: {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}
with Timer(unit="us", message="Micro benchmark"):
    pass
```

---