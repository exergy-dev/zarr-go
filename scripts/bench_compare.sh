#!/usr/bin/env bash
# Compare current benchmarks against the checked-in baseline.
# Fails (exit 1) if any geomean delta is worse than the threshold.
#
# Usage: ./scripts/bench_compare.sh [threshold_pct]
# Default threshold: 10% slowdown allowed (room for noise).
set -euo pipefail

THRESHOLD=${1:-10}
BASELINE=${BASELINE:-testdata/bench/baseline.txt}

if ! command -v benchstat >/dev/null; then
  echo "installing benchstat..." >&2
  go install golang.org/x/perf/cmd/benchstat@latest
fi

NEW=$(mktemp)
trap 'rm -f "$NEW"' EXIT
go test -bench=. -benchmem -run=^$ -count=5 . | grep -E "^(goos|goarch|pkg|cpu|Benchmark)" > "$NEW"

echo "=== benchstat ==="
benchstat "$BASELINE" "$NEW" | tee /tmp/benchstat.out

# Parse benchstat output for the geomean line; if any delta line shows a
# slowdown above THRESHOLD, fail.
python3 - "$THRESHOLD" <<'PY'
import re, sys
threshold = float(sys.argv[1])
worst = 0.0
worst_name = ""
with open("/tmp/benchstat.out") as f:
    for line in f:
        # Match lines like:  "BenchmarkX  100ns ± 1%  120ns ± 1%  +20.00% (p=...)"
        m = re.search(r"([+-]\d+\.\d+)%", line)
        if not m:
            continue
        delta = float(m.group(1))
        if delta > worst:
            worst = delta
            worst_name = line.strip()
if worst > threshold:
    print(f"REGRESSION: {worst:.1f}% > {threshold:.1f}% threshold")
    print(worst_name)
    sys.exit(1)
print(f"OK: max regression {worst:.1f}% (threshold {threshold:.1f}%)")
PY
