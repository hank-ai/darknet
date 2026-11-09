#!/usr/bin/env python3
"""Compare two `darknet detector test -ext_output` result files.

Usage: compare_detections.py NAME CPU.det GPU.det

Each .det file holds the detection lines, for example
    dog	c=83.726311%	x=137	y=206	w=181	h=332
The classes and boxes must be identical and the confidences must agree to
within 0.05 percentage points. The order of the detections is ignored.
"""
import re
import sys

LINE = re.compile(r"(.+?)\tc=([\d.]+)%\tx=(-?\d+)\ty=(-?\d+)\tw=(\d+)\th=(\d+)")


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = LINE.match(line.strip())
            if m:
                rows.append((m[1], float(m[2]), *map(int, m.groups()[2:])))
    return sorted(rows)


def main():
    name, cpu_path, gpu_path = sys.argv[1:4]
    cpu, gpu = load(cpu_path), load(gpu_path)
    ok = len(cpu) == len(gpu) > 0
    worst = 0.0
    for a, b in zip(cpu, gpu):
        ok &= a[0] == b[0] and a[2:] == b[2:]
        worst = max(worst, abs(a[1] - b[1]))
    ok &= worst < 0.05
    print(f"{name}: cpu={len(cpu)} gpu={len(gpu)} detections, "
          f"max confidence diff {worst:.6f} pct: {'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
