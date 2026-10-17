"""JSON (de)serialisation of schedules and small CSV helpers (resumable runs)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def sched_to_json(s: dict) -> dict:
    return dict(cmax=s["cmax"],
                ops={f"{o[0]},{o[1]}": v for o, v in s["ops"].items()},
                transports=s["transports"],
                machine_seq={str(k): [list(o) for o in v] for k, v in s["machine_seq"].items()},
                robot_seq={str(r): list(v) for r, v in s["robot_seq"].items()})


def sched_from_json(d: dict) -> dict:
    return dict(cmax=d["cmax"],
                ops={tuple(map(int, k.split(","))): v for k, v in d["ops"].items()},
                transports=d["transports"],
                machine_seq={int(k): [tuple(o) for o in v] for k, v in d["machine_seq"].items()},
                robot_seq={int(r): v for r, v in d["robot_seq"].items()})


def save_sched(path: Path, s: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sched_to_json(s)))


def load_sched(path: Path) -> dict:
    return sched_from_json(json.loads(Path(path).read_text()))


class CsvLog:
    """Append-only CSV with resume support (rows identified by key columns)."""

    def __init__(self, path: Path, key_cols):
        self.path, self.key_cols = Path(path), list(key_cols)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done, self.rows = set(), []
        if self.path.exists():
            with self.path.open(newline="") as f:
                for r in csv.DictReader(f):
                    self.rows.append(r)
                    self.done.add(self.key(r))

    def key(self, r):
        return tuple(str(r[c]) for c in self.key_cols)

    def has(self, **kv):
        return tuple(str(kv[c]) for c in self.key_cols) in self.done

    def get(self, **kv):
        k = tuple(str(kv[c]) for c in self.key_cols)
        for r in self.rows:
            if self.key(r) == k:
                return r
        return None

    def add(self, row: dict):
        new = not self.path.exists() or self.path.stat().st_size == 0
        if not new:
            with self.path.open(newline="") as f:
                header = next(csv.reader(f))
            extra = [c for c in row if c not in header]
            if extra:   # rewrite with the enlarged header
                rows = self.rows + [row]
                header = header + extra
                with self.path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=header)
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                self.rows.append({k: str(v) for k, v in row.items()})
                self.done.add(self.key(row))
                return
        else:
            header = list(row)
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        self.rows.append({k: str(v) for k, v in row.items()})
        self.done.add(self.key(row))
