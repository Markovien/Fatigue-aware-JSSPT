"""Warm start for CP-SAT: best of five dispatching-rule schedules (feasible for the model)."""
import random

from .decoder import Decoder
from .metaheuristics import seeds


def warm_start(inst, cfg):
    dec = Decoder(inst, cfg)
    best = None
    for q in seeds(dec, random.Random(0), 5):
        c, st = dec.decode(q, record=True)
        if best is None or c < best[0]:
            best = (c, st)
    return dec.to_schedule(best[1], best[0]), best[0]
