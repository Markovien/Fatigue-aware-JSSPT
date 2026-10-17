"""Self-check (about 2 min): CP-SAT, decoder, validator and replay agree.
    python tests/self_check.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fajsspt import instances as I                       # noqa: E402
from fajsspt.cpsat_model import solve                      # noqa: E402
from fajsspt.decoder import Decoder                        # noqa: E402
from fajsspt.fatigue import FatigueConfig                  # noqa: E402
from fajsspt.replay import replay, validate                # noqa: E402
from fajsspt.warm import warm_start                        # noqa: E402

PAPER = {"EX11": (98, 106), "EX52": (63, 73)}             # submitted OPT1 / S1 CP-SAT
ok = True
for name, (o1, o2) in PAPER.items():
    inst = I.bu_adapted(int(name[2:-1]), int(name[-1]))
    r0 = solve(inst, FatigueConfig(use_fatigue=False), time_limit=60)
    r1 = solve(inst, FatigueConfig(hard_bound=False), time_limit=120)
    print(f"{name}: OPT1 {r0['cmax']} (paper {o1}), legacy OPT2 {r1['cmax']} (paper {o2})")
    ok &= r0["cmax"] == o1 and r1["cmax"] == o2
    for cfg in (FatigueConfig(), FatigueConfig(recover_on_idle=True, idle_factor=0.25),
                FatigueConfig(rounding="conservative")):
        h, ub = warm_start(inst, cfg)
        r = solve(inst, cfg, time_limit=120, hint=h, ub=ub)
        e = validate(inst, r["schedule"], cfg)
        rp = replay(inst, r["schedule"], cfg)
        dec = Decoder(inst, cfg)
        bad = 0
        for _ in range(100):
            q = dec.base_seq[:]
            random.shuffle(q)
            c, st = dec.decode(q, record=True)
            bad += bool(validate(inst, dec.to_schedule(st, c), cfg))
        print(f"   {cfg.label()}: CP-SAT {r['status']} {r['cmax']} | validation errors {len(e)} | "
              f"replay {rp['cmax']} | invalid decoded schedules {bad}/100")
        ok &= not e and rp["cmax"] == r["cmax"] and bad == 0
print("SELF-CHECK", "PASSED" if ok else "FAILED")
