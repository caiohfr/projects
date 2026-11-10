# Roadload performance baseline v0.1

**Commit:** `fca9a9ad189fe7e6930e987e29fee5717c0d532e`  
**Executed:** 2026-11-09 UTC  
**Environment:** Windows 10 build 26100; Python 3.11.9; NumPy 2.3.3; 12 logical CPUs; Intel64 Family 6 Model 154. RAM was unavailable to the non-elevated probe.  
**Protocol:** synthetic in-process inputs, one warm-up, 20 measured repetitions, `perf_counter_ns`, `process_time_ns`, and a separate `tracemalloc` sample.

| Operation | Workload | p50 wall ms | p95 wall ms | p50 CPU ms | p95 CPU ms | traced peak bytes |
|---|---:|---:|---:|---:|---:|---:|
| Roadload force scalar | 100 km/h | 0.0011 | 0.0032 | 0.000 | 0.000 | 48 |
| Roadload force vector | n=10,000 | 0.0223 | 0.0229 | 0.000 | 0.000 | 240,392 |
| Roadload power vector | n=10,000 | 0.0314 | 0.0325 | 0.000 | 0.000 | 320,488 |
| Canonical VDE series | n=10,000 | 0.0752 | 0.2129 | 0.000 | 0.000 | 480,896 |
| Roadload model batch | 1,000 vehicles | 0.1472 | 0.1573 | 0.000 | 0.000 | 30,696 |
| Legacy scenario batch | 1,000 vehicles | 8.8054 | 23.9499 | 15.625 | 15.625 | 1,564,624 |

CPU values below the Windows timer resolution appear as zero and must not be interpreted as zero cost. The local JSON also contains n=100/1,000 vectors and 1/10/100 vehicle batches.

## Explicit gaps

- SQL query count: **NOT MEASURED**; the benchmark does not enter repositories.
- Streamlit/browser latency: **NOT MEASURED**.
- Full-process RSS: **NOT MEASURED**; `tracemalloc` is Python traced allocation only.
- Component breakdown n×k: **NOT AVAILABLE** in the current model; no mock loop was presented as production behavior.
- Cold-start/import/cache variance: **NOT MEASURED**.

This baseline supports later before/after comparison of the listed pure paths, not a whole-application performance claim.
