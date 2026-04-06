import argparse
import ast
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Travel times
# ---------------------------------------------------------------------------

TRAVEL_TIME_MATRIX = {
    # L/U, M1, M2, M3, M4
    1: [[0, 6, 8, 8, 6],
        [6, 0, 6, 8, 10],
        [8, 6, 0, 6, 8],
        [8, 8, 6, 0, 6],
        [6, 10, 8, 6, 0]],

    2: [[0, 4, 6, 6, 4],
        [4, 0, 2, 4, 2],
        [6, 2, 0, 2, 4],
        [6, 4, 2, 0, 2],
        [4, 2, 4, 2, 0]],

    3: [[0, 2, 4, 4, 2],
        [2, 0, 2, 6, 4],
        [4, 2, 0, 6, 6],
        [4, 6, 6, 0, 2],
        [2, 4, 6, 2, 0]],

    4: [[0, 4, 8, 10, 14],
        [4, 0, 4, 6, 10],
        [8, 4, 0, 6, 6],
        [10, 6, 6, 0, 6],
        [14, 10, 6, 6, 0]],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_half_up(value: float) -> int:
    """Round a non-negative float to the nearest integer using half-up logic."""
    if value < 0:
        raise ValueError(f"Negative durations are not supported: {value}")
    return int(math.floor(value + 0.5))


def fatigue_after_processing(F_start: float, duration: float, lam: float) -> float:
    if lam <= 0 or duration <= 0:
        return F_start
    return 1.0 - (1.0 - F_start) * math.exp(-lam * duration)


def fatigue_work_curve(F_start: float, t0: float, t1: float, lam: float, num: int = 25):
    if t1 <= t0:
        return [t0, t1], [F_start, F_start]
    step = (t1 - t0) / max(1, num - 1)
    times = [t0 + i * step for i in range(num)]
    if lam <= 0:
        values = [F_start for _ in times]
    else:
        values = [1.0 - (1.0 - F_start) * math.exp(-lam * (t - t0)) for t in times]
    return times, values


def fatigue_rest_curve(F_start: float, F_min: float, t0: float, t1: float, mu: float, num: int = 25):
    if t1 <= t0:
        return [t0, t1], [F_start, F_start]
    step = (t1 - t0) / max(1, num - 1)
    times = [t0 + i * step for i in range(num)]
    if mu <= 0:
        values = [F_start for _ in times]
    else:
        values = [max(F_min, F_start * math.exp(-mu * (t - t0))) for t in times]
    return times, values


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MachineTaskSpec:
    machine_id: int
    job_id: int
    planned_start: int
    planned_end: int
    op_idx: int

    @property
    def nominal_duration(self) -> int:
        return self.planned_end - self.planned_start

    @property
    def node_id(self) -> str:
        return f"O:{self.job_id}:{self.op_idx}"


@dataclass(frozen=True)
class RobotTaskSpec:
    robot_id: int
    job_id: int
    planned_start: int
    planned_end: int
    op_idx: int  # 99 means final return to depot

    @property
    def nominal_duration(self) -> int:
        return self.planned_end - self.planned_start

    @property
    def node_id(self) -> str:
        return f"T:{self.job_id}:{self.op_idx}"


@dataclass
class ExecutedTask:
    node_id: str
    resource_type: str
    resource_id: int
    job_id: int
    op_idx: int
    planned_start: int
    planned_end: int
    actual_start: int = 0
    actual_end: int = 0
    duration: int = 0
    nominal_duration: int = 0
    kind: str = "activity"  # activity or rest
    label: str = ""
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class FatigueSegment:
    machine_id: int
    start: int
    end: int
    F_start: float
    F_end: float
    segment_type: str  # processing / rest / idle
    job_id: Optional[int] = None
    op_idx: Optional[int] = None


@dataclass
class MachineState:
    fatigue: float
    last_end: int = 0


@dataclass
class SimulationResult:
    instance_id: str
    mode: str
    makespan: int
    machine_tasks: List[ExecutedTask]
    robot_tasks: List[ExecutedTask]
    rest_tasks: List[ExecutedTask]
    fatigue_log: List[FatigueSegment]
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_schedule_blocks(path: str | Path) -> Dict[str, List[List[int]]]:
    text = Path(path).read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in text.split("-") if chunk.strip()]
    data: Dict[str, List[List[int]]] = {}
    for chunk in chunks:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        inst_id = lines[0]
        records = ast.literal_eval(lines[1])
        data[inst_id] = records
    return data


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class ScheduleSimulator:
    def __init__(
        self,
        machine_schedule_path: str | Path,
        robot_schedule_path: str | Path,
        *,
        lam: float = 0.04,
        mu: float = 0.08,
        delta: float = (0.3/0.04) * math.log(2.0),
        F_max: float = 0.8,
        F_min: float = 0.2,
        discretization_steps: int = 1000,
        max_fatigue_level: int = 1000
    ) -> None:
        self.machine_raw = parse_schedule_blocks(machine_schedule_path)
        self.robot_raw = parse_schedule_blocks(robot_schedule_path)
        self.instance_ids = sorted(set(self.machine_raw) & set(self.robot_raw), key=self._instance_sort_key)
        self.lam = lam
        self.mu = mu
        self.delta = delta
        self.F_max = F_max
        self.F_min = F_min

        # Discretization parameters
        self.DISCRETIZATION_STEPS = discretization_steps
        self.MAX_FATIGUE_LEVEL = max_fatigue_level
        self.FATIGUE_UNIT = float(self.MAX_FATIGUE_LEVEL) / self.DISCRETIZATION_STEPS

        self.processing_fatigue_lookup: Dict[int, float] = {}
        self.recovery_fatigue_lookup: Dict[int, float] = {}
        self._generate_fatigue_lookups()

    @staticmethod
    def _instance_sort_key(inst_id: str) -> Tuple[int, str]:
        digits = "".join(ch for ch in inst_id if ch.isdigit())
        return (int(digits) if digits else 10**9, inst_id)

    @staticmethod
    def _instance_matrix_key(instance_id: str) -> int:
        """
        EX11 -> 1, EX12 -> 2, ..., EX104 -> 4
        """
        digits = "".join(ch for ch in instance_id if ch.isdigit())
        if not digits:
            raise ValueError(f"Cannot decode matrix family from instance id: {instance_id}")
        return int(digits[-1])

    def _travel_time(self, instance_id: str, from_node: int, to_node: int) -> int:
        key = self._instance_matrix_key(instance_id)
        return int(TRAVEL_TIME_MATRIX[key][from_node][to_node])

    @staticmethod
    def _robot_task_nodes(
        machine_by_key: Dict[Tuple[int, int], MachineTaskSpec],
        job_last_op: Dict[int, int],
        job_id: int,
        op_idx: int,
    ) -> Tuple[int, int]:
        """
        Return (pickup_node, drop_node) for a robot task.

        Node convention:
        0 = depot (L/U)
        1..4 = machine ids M1..M4
        """
        if op_idx == 1:
            pickup_node = 0
            drop_node = machine_by_key[(job_id, 1)].machine_id
        elif op_idx == 99:
            last_op = job_last_op[job_id]
            pickup_node = machine_by_key[(job_id, last_op)].machine_id
            drop_node = 0
        else:
            pickup_node = machine_by_key[(job_id, op_idx - 1)].machine_id
            drop_node = machine_by_key[(job_id, op_idx)].machine_id

        return pickup_node, drop_node

    def list_instances(self) -> List[str]:
        return list(self.instance_ids)

    def load_instance(self, instance_id: str) -> Tuple[List[MachineTaskSpec], List[RobotTaskSpec]]:
        if instance_id not in self.machine_raw:
            raise KeyError(f"Instance {instance_id} not found in machine schedule file.")
        if instance_id not in self.robot_raw:
            raise KeyError(f"Instance {instance_id} not found in robot schedule file.")

        machine_tasks = [MachineTaskSpec(*row) for row in self.machine_raw[instance_id]]
        robot_tasks = [RobotTaskSpec(*row) for row in self.robot_raw[instance_id]]
        return machine_tasks, robot_tasks

    def _generate_fatigue_lookups(self) -> None:
        """Generates lookup tables for fatigue and recovery durations."""
        for f_level in range(self.MAX_FATIGUE_LEVEL + 1):
            f_value = f_level / self.DISCRETIZATION_STEPS

            # Processing Fatigue Lookups (effective processing time multiplier)
            # Equivalent to: (1.0 + self.delta * self.lam * math.log(1.0 + f_value))
            # Storing the multiplier directly
            self.processing_fatigue_lookup[f_level] = 1.0 + self.delta * self.lam * math.log(1.0 + max(f_value, -0.999999))

            # Recovery Fatigue Lookups (duration to recover from f_value to F_min)
            # Equivalent to: math.log(f_value / self.F_min) / self.mu
            if self.mu > 0 and f_value > self.F_min + 1e-9:
                self.recovery_fatigue_lookup[f_level] = int(math.ceil(math.log(f_value / self.F_min) / self.mu))
            else:
                self.recovery_fatigue_lookup[f_level] = 0.0

    def _fatigue_to_level(self, fatigue_value: float) -> int:
        """Converts a continuous fatigue value to its discretized integer level."""
        return round_half_up(fatigue_value * self.DISCRETIZATION_STEPS)

    def _level_to_fatigue(self, fatigue_level: int) -> float:
        """Converts a discretized fatigue level to its continuous value."""
        return float(fatigue_level) / self.DISCRETIZATION_STEPS

    def simulate_instance(self, instance_id: str, *, use_fatigue: bool = False) -> SimulationResult:
        machine_specs, robot_specs = self.load_instance(instance_id)

        machine_by_key: Dict[Tuple[int, int], MachineTaskSpec] = {(x.job_id, x.op_idx): x for x in machine_specs}
        robot_by_key: Dict[Tuple[int, int], RobotTaskSpec] = {(x.job_id, x.op_idx): x for x in robot_specs}

        # Resource sequences are fixed by the provided schedules.
        machine_seq: Dict[int, List[MachineTaskSpec]] = defaultdict(list)
        for spec in sorted(machine_specs, key=lambda x: (x.machine_id, x.planned_start, x.job_id, x.op_idx)):
            machine_seq[spec.machine_id].append(spec)

        robot_seq: Dict[int, List[RobotTaskSpec]] = defaultdict(list)
        for spec in sorted(robot_specs, key=lambda x: (x.robot_id, x.planned_start, x.job_id, x.op_idx)):
            robot_seq[spec.robot_id].append(spec)

        # Determine the last operation for each machine
        last_op_on_machine: Dict[int, MachineTaskSpec] = {}
        for m_id, seq in machine_seq.items():
            if seq:
                last_op_on_machine[m_id] = seq[-1]

        # Build precedence graph: fixed resource order + job flow.
        node_ids = {spec.node_id for spec in machine_specs} | {spec.node_id for spec in robot_specs}
        preds: Dict[str, set[str]] = {nid: set() for nid in node_ids}
        succs: Dict[str, set[str]] = {nid: set() for nid in node_ids}

        def add_edge(u: str, v: str) -> None:
            if v not in succs[u]:
                succs[u].add(v)
                preds[v].add(u)

        for seq in machine_seq.values():
            for prev, cur in zip(seq, seq[1:]):
                add_edge(prev.node_id, cur.node_id)

        for seq in robot_seq.values():
            for prev, cur in zip(seq, seq[1:]):
                add_edge(prev.node_id, cur.node_id)

        job_last_op: Dict[int, int] = defaultdict(int)
        for spec in machine_specs:
            job_last_op[spec.job_id] = max(job_last_op[spec.job_id], spec.op_idx)

        for job_id, last_op in job_last_op.items():
            for op_idx in range(1, last_op + 1):
                transport = robot_by_key.get((job_id, op_idx))
                machine = machine_by_key.get((job_id, op_idx))
                if transport is not None and machine is not None:
                    add_edge(transport.node_id, machine.node_id)
                if op_idx < last_op:
                    next_transport = robot_by_key.get((job_id, op_idx + 1))
                    if machine is not None and next_transport is not None:
                        add_edge(machine.node_id, next_transport.node_id)
                else:
                    final_transport = robot_by_key.get((job_id, 99))
                    if machine is not None and final_transport is not None:
                        add_edge(machine.node_id, final_transport.node_id)

        # Topological order.
        indeg = {nid: len(preds[nid]) for nid in node_ids}
        ready = [nid for nid in node_ids if indeg[nid] == 0]
        ready.sort(key=lambda nid: self._planned_start_of(nid, machine_by_key, robot_by_key))
        topo: List[str] = []
        dq = deque(ready)
        while dq:
            nid = dq.popleft()
            topo.append(nid)
            for child in sorted(succs[nid], key=lambda z: self._planned_start_of(z, machine_by_key, robot_by_key)):
                indeg[child] -= 1
                if indeg[child] == 0:
                    dq.append(child)

        if len(topo) != len(node_ids):
            raise ValueError(f"Instance {instance_id} contains cyclic precedence/resource constraints.")

        executed: Dict[str, ExecutedTask] = {}
        fatigue_log: List[FatigueSegment] = []
        robot_last_end: Dict[int, int] = {r_id: 0 for r_id in robot_seq.keys()}
        robot_last_drop: Dict[int, int] = {r_id: 0 for r_id in robot_seq.keys()}  # all robots start at depot

        # Initialize machine state with discrete fatigue levels
        F_min_level = self._fatigue_to_level(self.F_min)
        machine_state: Dict[int, MachineState] = {
            m_id: MachineState(fatigue=F_min_level, last_end=0) for m_id in machine_seq.keys()
        }
        fatigue_violations = 0

        for nid in topo:
            pred_end = 0
            for p in preds[nid]:
                pred_end = max(pred_end, executed[p].actual_end)

            if nid.startswith("T:"):
                job_id, op_idx = self._decode_node(nid)
                spec = robot_by_key[(job_id, op_idx)]

                pickup_node, drop_node = self._robot_task_nodes(
                    machine_by_key, job_last_op, job_id, op_idx
                )

                prev_drop = robot_last_drop[spec.robot_id]
                empty_move = self._travel_time(instance_id, prev_drop, pickup_node)
                robot_ready = robot_last_end[spec.robot_id] + empty_move

                # A robot task can start only when:
                # - all precedence constraints are satisfied,
                # - the robot is available,
                # - and the robot has had enough time to travel from the
                #   previous drop location to the current pickup location.
                start = max(spec.planned_start, pred_end, robot_ready)
                end = start + spec.nominal_duration

                label = f"T^R_{job_id}" if op_idx == 99 else f"T(O{job_id},{op_idx})"
                executed[nid] = ExecutedTask(
                    node_id=nid,
                    resource_type="robot",
                    resource_id=spec.robot_id,
                    job_id=job_id,
                    op_idx=op_idx,
                    planned_start=spec.planned_start,
                    planned_end=spec.planned_end,
                    actual_start=start,
                    actual_end=end,
                    duration=spec.nominal_duration,
                    nominal_duration=spec.nominal_duration,
                    kind="activity",
                    label=label,
                    metadata={
                        "pickup_node": pickup_node,
                        "drop_node": drop_node,
                        "prev_drop": prev_drop,
                        "empty_move": empty_move,
                        "robot_ready": robot_ready,
                    },
                )

                robot_last_end[spec.robot_id] = end
                robot_last_drop[spec.robot_id] = drop_node
                continue

            # Machine task
            job_id, op_idx = self._decode_node(nid)
            spec = machine_by_key[(job_id, op_idx)]
            state = machine_state[spec.machine_id]
            #release = max(spec.planned_start, pred_end)
            release = max(spec.planned_start, pred_end, state.last_end)

            if release > state.last_end:
                # Record idle time, fatigue doesn't change during idle time in this model
                fatigue_log.append(
                    FatigueSegment(
                        machine_id=spec.machine_id,
                        start=state.last_end,
                        end=release,
                        F_start=self._level_to_fatigue(state.fatigue),
                        F_end=self._level_to_fatigue(state.fatigue),
                        segment_type="idle",
                    )
                )

            rest_tasks: List[ExecutedTask] = []
            op_start = release
            F_before_level = state.fatigue # Discrete fatigue level

            if use_fatigue:
                iter_guard = 0
                while True:
                    iter_guard += 1
                    if iter_guard > 3:
                        # Safety: avoid infinite cycle; accept and cap fatigue.
                        break

                    # Convert discrete level to continuous value for comparison with F_max
                    F_before_cont = self._level_to_fatigue(F_before_level)

                    # Check F_before < F_max
                    if F_before_cont >= self.F_max - 1e-9:
                        fatigue_violations += 1
                        op_start, F_before_level = self._force_rest(spec.machine_id, job_id, op_idx, op_start, F_before_level, rest_tasks, fatigue_log)
                        continue # Re-evaluate with new F_before

                    # Calculate effective processing time using lookup table
                    # Clamp fatigue level to MAX_FATIGUE_LEVEL
                    clamped_F_before_level = min(F_before_level, self.MAX_FATIGUE_LEVEL)
                    proc_time_multiplier = self.processing_fatigue_lookup[clamped_F_before_level]

                    inflated_duration = spec.nominal_duration * proc_time_multiplier
                    proc_dur = round_half_up(max(0.0, inflated_duration))

                    # Calculate F_after using continuous function for accuracy, then discretize
                    F_after_cont = fatigue_after_processing(F_before_cont, proc_dur, self.lam)
                    F_after_level = self._fatigue_to_level(F_after_cont)
                    F_after_level = min(F_after_level, self.MAX_FATIGUE_LEVEL) # Clamp max fatigue level

                    # Check F_after < 1 (effectively F_after_level < MAX_FATIGUE_LEVEL)
                    if F_after_level >= self.MAX_FATIGUE_LEVEL - 1e-9:
                         # If F_before is already at F_min and F_after still exceeds 1, something is wrong with parameters.
                        if F_before_cont <= self.F_min + 1e-12:
                            raise ValueError(
                                f"Instance {instance_id}: operation O({job_id},{op_idx}) on M{spec.machine_id} "
                                f"still violates F_after < 1 even when starting at F_min."
                            )
                        op_start, F_before_level = self._force_rest(spec.machine_id, job_id, op_idx, op_start, F_before_level, rest_tasks, fatigue_log)
                        continue # Re-evaluate with new F_before

                    # If both checks pass, we can break from the rest loop
                    break

            else:
                proc_dur = spec.nominal_duration
                F_before_cont = self._level_to_fatigue(F_before_level)
                if F_before_cont >= self.F_max - 1e-9: fatigue_violations += 1
                F_after_cont = fatigue_after_processing(F_before_cont, proc_dur, self.lam)
                F_after_level = self._fatigue_to_level(F_after_cont) # Assign F_after_level here

            op_end = op_start + proc_dur
            op_F_end_level = F_after_level
            op_F_end_cont = self._level_to_fatigue(op_F_end_level) if use_fatigue else F_after_cont

            fatigue_log.append(
                FatigueSegment(
                    machine_id=spec.machine_id,
                    start=op_start,
                    end=op_end,
                    F_start=self._level_to_fatigue(F_before_level),
                    F_end=op_F_end_cont,
                    segment_type="processing",
                    job_id=job_id,
                    op_idx=op_idx,
                )
            )

            # Store the operation FIRST, using the true operation end.
            executed[nid] = ExecutedTask(
                node_id=nid,
                resource_type="machine",
                resource_id=spec.machine_id,
                job_id=job_id,
                op_idx=op_idx,
                planned_start=spec.planned_start,
                planned_end=spec.planned_end,
                actual_start=op_start,
                actual_end=op_end,
                duration=proc_dur,
                nominal_duration=spec.nominal_duration,
                kind="activity",
                label=f"O({job_id},{op_idx})",
                metadata={
                    "F_start": self._level_to_fatigue(F_before_level),
                    "F_end": op_F_end_cont,
                    "release": release,
                },
            )

            # Force rest after operation completion if needed
            is_last_op = (spec == last_op_on_machine.get(spec.machine_id))
            if use_fatigue and op_F_end_level >= self._fatigue_to_level(self.F_max) - 1e-9 and not is_last_op:
                rest_after_op_start = op_end
                rest_after_op_F_before_level = op_F_end_level

                rest_after_op_end, rest_after_op_F_after_level = self._force_rest(
                    spec.machine_id,
                    job_id,
                    op_idx,
                    rest_after_op_start,
                    rest_after_op_F_before_level,
                    rest_tasks,
                    fatigue_log,
                )

                state.fatigue = rest_after_op_F_after_level
                state.last_end = rest_after_op_end
            else:
                state.fatigue = op_F_end_level
                state.last_end = op_end

            for rest_task in rest_tasks:
                executed[rest_task.node_id] = rest_task

        machine_tasks = [x for x in executed.values() if x.resource_type == "machine" and x.kind == "activity"]
        robot_tasks = [x for x in executed.values() if x.resource_type == "robot"]
        rest_tasks = [x for x in executed.values() if x.kind == "rest"]
        machine_tasks.sort(key=lambda x: (x.resource_id, x.actual_start, x.job_id, x.op_idx))
        robot_tasks.sort(key=lambda x: (x.resource_id, x.actual_start, x.job_id, x.op_idx))
        rest_tasks.sort(key=lambda x: (x.resource_id, x.actual_start, x.job_id, x.op_idx))
        fatigue_log.sort(key=lambda x: (x.machine_id, x.start, x.segment_type))

        makespan = max((task.actual_end for task in list(machine_tasks) + list(robot_tasks) + list(rest_tasks)), default=0)
        metrics = self._compute_metrics(makespan, machine_tasks, robot_tasks, rest_tasks, fatigue_log, fatigue_violations)

        return SimulationResult(
            instance_id=instance_id,
            mode="fatigue_on" if use_fatigue else "fatigue_off",
            makespan=makespan,
            machine_tasks=machine_tasks,
            robot_tasks=robot_tasks,
            rest_tasks=rest_tasks,
            fatigue_log=fatigue_log,
            metrics=metrics,
        )

    def _force_rest(self, machine_id: int, job_id: int, op_idx: int, current_time: int, current_F_level: int, rest_tasks_list: List[ExecutedTask], fatigue_log_list: List[FatigueSegment]) -> Tuple[int, int]:
        """Helper to force a machine to rest until F_min (discretized)."""
        target_F_level = self._fatigue_to_level(self.F_min)

        # Lookup rest duration from current F_level to F_min_level
        # Clamp fatigue level to MAX_FATIGUE_LEVEL
        clamped_F_level = min(current_F_level, self.MAX_FATIGUE_LEVEL)
        rest_exact = self.recovery_fatigue_lookup.get(clamped_F_level, 0.0)

        rest_dur = round_half_up(max(0.0, rest_exact))
        rest_end = current_time + rest_dur

        fatigue_log_list.append(
            FatigueSegment(
                machine_id=machine_id,
                start=current_time,
                end=rest_end,
                F_start=self._level_to_fatigue(current_F_level),
                F_end=self._level_to_fatigue(target_F_level),
                segment_type="rest",
            )
        )
        rest_tasks_list.append(
            ExecutedTask(
                node_id=f"REST:{machine_id}:{job_id}:{op_idx}:{len(rest_tasks_list)+1}",
                resource_type="machine",
                resource_id=machine_id,
                job_id=job_id,
                op_idx=op_idx,
                planned_start=current_time,
                planned_end=rest_end,
                actual_start=current_time,
                actual_end=rest_end,
                duration=rest_dur,
                nominal_duration=rest_dur,
                kind="rest",
                label="REST",
                metadata={
                    "F_start": self._level_to_fatigue(current_F_level),
                    "F_end": self._level_to_fatigue(target_F_level)
                },
            )
        )
        return rest_end, target_F_level

    def simulate_all(self, *, use_fatigue: bool = False) -> Dict[str, SimulationResult]:
        return {inst_id: self.simulate_instance(inst_id, use_fatigue=use_fatigue) for inst_id in self.instance_ids}

    @staticmethod
    def _decode_node(node_id: str) -> Tuple[int, int]:
        _, job_id, op_idx = node_id.split(":")
        return int(job_id), int(op_idx)

    @staticmethod
    def _planned_start_of(
        node_id: str,
        machine_by_key: Dict[Tuple[int, int], MachineTaskSpec],
        robot_by_key: Dict[Tuple[int, int], RobotTaskSpec],
    ) -> int:
        job_id, op_idx = ScheduleSimulator._decode_node(node_id)
        if node_id.startswith("O:"):
            return machine_by_key[(job_id, op_idx)].planned_start
        return robot_by_key[(job_id, op_idx)].planned_start

    def _compute_metrics(
        self,
        makespan: int,
        machine_tasks: List[ExecutedTask],
        robot_tasks: List[ExecutedTask],
        rest_tasks: List[ExecutedTask],
        fatigue_log: List[FatigueSegment],
        fatigue_violations = 0,
    ) -> Dict[str, float]:
        machine_busy = defaultdict(int)
        robot_busy = defaultdict(int)
        machine_rest = defaultdict(int)

        for task in machine_tasks:
            machine_busy[task.resource_id] += task.duration
        for task in robot_tasks:
            robot_busy[task.resource_id] += task.duration
        for task in rest_tasks:
            machine_rest[task.resource_id] += task.duration

        # Use continuous values from fatigue_log for metrics
        fatigue_max = max((max(seg.F_start, seg.F_end) for seg in fatigue_log), default=self.F_min)
        fatigue_growth = sum(max(0.0, seg.F_end - seg.F_start) for seg in fatigue_log if seg.segment_type == "processing")
        fatigue_acc = sum(seg.F_start * (seg.end - seg.start) for seg in fatigue_log if seg.segment_type == "processing")

        metrics: Dict[str, float] = {
            "makespan": makespan,
            "fatigue_violations": fatigue_violations,
            "fatigue_max": fatigue_max,
            "fatigue_growth": fatigue_growth,
            "fatigue_acc": fatigue_acc,
            "total_rest_time": sum(task.duration for task in rest_tasks),
            "num_machine_tasks": len(machine_tasks),
            "num_robot_tasks": len(robot_tasks),
            "num_rest_blocks": len(rest_tasks),
            
        }

        for m_id in sorted(set([task.resource_id for task in machine_tasks] + [task.resource_id for task in rest_tasks])):
            metrics[f"machine_{m_id}_busy"] = machine_busy[m_id]
            metrics[f"machine_{m_id}_rest"] = machine_rest[m_id]
            metrics[f"machine_{m_id}_util"] = machine_busy[m_id] / makespan if makespan > 0 else 0.0

        for r_id in sorted(set(task.resource_id for task in robot_tasks)):
            metrics[f"robot_{r_id}_busy"] = robot_busy[r_id]
            metrics[f"robot_{r_id}_util"] = robot_busy[r_id] / makespan if makespan > 0 else 0.0

        return metrics

    # ---------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------

    def gantt_generator(self, result: SimulationResult, output_path: str | Path, plot_gantt: bool = False, save_gantt: bool = False, use_fatigue: bool = False, *, title: Optional[str] = None) -> None:
        machine_ids = sorted({task.resource_id for task in result.machine_tasks} | {task.resource_id for task in result.rest_tasks})
        robot_ids = sorted({task.resource_id for task in result.robot_tasks})
        colors = [
            "tab:red", "tab:cyan", "tab:green", "tab:orange", "gold",
            "tab:brown", "magenta", "lime", "tomato", "tab:blue",
            "slateblue", "orchid", "darkseagreen", "coral",
        ]

        fig, (ax_gantt, ax_fat) = plt.subplots(
            2, 1, figsize=(20, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        yticks_positions: List[int] = [] # Actual positions for the bars
        yticks_labels: List[str] = [] # Positions for the labels
        ylabels: List[str] = [] # The labels themselves

        # Use 1-based indexing for plotting
        current_y_level = 0

        for m_id in machine_ids:
            current_y_level += 1
            yticks_positions.append(10 * current_y_level) # e.g., 10, 20, 30
            yticks_labels.append(10 * current_y_level + 5) # e.g., 15, 25, 35
            ylabels.append(f"M{m_id}")

        for r_id in robot_ids:
            current_y_level += 1
            yticks_positions.append(10 * current_y_level) # e.g., 50, 60
            yticks_labels.append(10 * current_y_level + 5) # e.g., 55, 65
            ylabels.append(f"R{r_id}")

        ax_gantt.set_yticks(yticks_labels)
        ax_gantt.set_yticklabels(ylabels, fontsize=16)
        ax_gantt.set_xlabel("Time", fontsize=18, weight="bold")
        ax_gantt.set_ylabel("", fontsize=18, weight="bold")
        ax_gantt.tick_params(labelsize=14)
        ax_gantt.minorticks_on()
        ax_gantt.grid(which="major", linestyle="-", linewidth=0.5, color="red")
        ax_gantt.grid(which="minor", linestyle=":", linewidth=0.5, color="black")
        ax_gantt.set_xlim(0, result.makespan + 10)

        # Create a mapping for y-positions based on resource ID and type
        y_level_map = {}
        k_idx = 0
        for m_id in machine_ids:
            y_level_map[('M', m_id)] = yticks_positions[k_idx]
            k_idx += 1
        for r_id in robot_ids:
            y_level_map[('R', r_id)] = yticks_positions[k_idx]
            k_idx += 1


        # Draw machine tasks and rests.
        activities = sorted(
            list(result.machine_tasks) + list(result.rest_tasks),
            key=lambda x: (x.resource_id, x.actual_start, 0 if x.kind == "rest" else 1),
        )
        for task in activities:
            y_base = y_level_map[('M', task.resource_id)]
            width = task.actual_end - task.actual_start
            if task.kind == "rest":
                ax_gantt.broken_barh(
                    [(task.actual_start, width)],
                    (y_base, 9), # Bar height is 9
                    facecolors="lightgray",
                    #edgecolors="black",
                    #hatch="//",
                    linewidth=0.5
                )
                fs = task.metadata.get("F_start", 0.0)
                fe = task.metadata.get("F_end", 0.0)
                ax_gantt.text(task.actual_start + width / 2, 4.5 + y_base, f"Rest\n{fs:.2f}→{fe:.2f}", ha="center", va="center", fontsize=12)
            else:
                color = colors[(task.job_id - 1) % len(colors)]
                ax_gantt.broken_barh([(task.actual_start, width)], (y_base, 9), facecolors=color) # Bar height is 9
                ax_gantt.text(task.actual_start + width / 2, 4.5 + y_base, r"$O_{%d,%d}$" % (task.job_id, task.op_idx), ha="center", va="center", fontsize=14, weight="bold")

        for task in result.robot_tasks:
            y_base = y_level_map[('R', task.resource_id)]
            width = task.actual_end - task.actual_start
            color = colors[(task.job_id - 1) % len(colors)]
            ax_gantt.broken_barh([(task.actual_start, width)], (y_base, 9), facecolors=color) # Bar height is 9

            if task.op_idx == 99:
                text_label = r"$T^R_{J%d}$" % task.job_id
            else:
                text_label = r"$T_{%d,%d}$" % (task.job_id, task.op_idx)

            ax_gantt.text(task.actual_start + width / 2, 4.5 + y_base, text_label, ha="center", va="center", fontsize=14, weight="bold")

        # Title
        resolution_method = "Simulator + Fatigue" if use_fatigue else "Simulator (no fatigue)"
        chart_title = f"{result.instance_id}\n{resolution_method}, Cmax={int(round(result.makespan))}"
        ax_gantt.set_title(title or chart_title, fontsize=20, weight="bold")

        # Fatigue subplot
        ax_fat.set_xlabel("Time", fontsize=16, weight="bold")
        ax_fat.set_ylabel("Fatigue", fontsize=16, weight="bold")
        ax_fat.set_ylim(-0.05, 1.05)
        ax_fat.grid(True, linestyle=":", alpha=0.7)
        ax_fat.tick_params(labelsize=14)

        for idx, m_id in enumerate(machine_ids):
            m_color = colors[idx % len(colors)]
            segments = [seg for seg in result.fatigue_log if seg.machine_id == m_id]
            segments.sort(key=lambda x: (x.start, x.end, x.segment_type))
            first = True
            for seg in segments:
                if seg.segment_type == "idle":
                    ax_fat.plot([seg.start, seg.end], [seg.F_start, seg.F_end], color=m_color, linewidth=2, alpha=0.75, label=f"M{m_id}" if first else None)
                elif seg.segment_type == "rest":
                    tx, fy = fatigue_rest_curve(seg.F_start, self.F_min, seg.start, seg.end, self.mu)
                    ax_fat.plot(tx, fy, color=m_color, linewidth=2, alpha=0.85, label=f"M{m_id}" if first else None)
                else:
                    tx, fy = fatigue_work_curve(seg.F_start, seg.start, seg.end, self.lam)
                    ax_fat.plot(tx, fy, color=m_color, linewidth=2, alpha=0.85, label=f"M{m_id}" if first else None)
                first = False

        ax_fat.legend(loc="upper right")
        ax_gantt.set_xlim(0, result.makespan + 10)
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if plot_gantt == True: plt.show()
        if save_gantt == True:
            fig.savefig(output_path, dpi=300)
            print(f"Gantt saved to: {output_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

# Paths to input files
machine_schedule_path = Path("machine_schedules.txt")
robot_schedule_path = Path("robot_schedules.txt")

# Simulator creation
simulator = ScheduleSimulator(
    machine_schedule_path=machine_schedule_path,
    robot_schedule_path=robot_schedule_path,
)

# Choose the instance and simulation mode
instance_id = "EX82"
use_fatigue = True   # True = fatigue-on: rest are enforced when needed, False = fatigue-off: fatigue tracking simply
plot_gantt = True
save_gantt = False

# Run the simulation
result = simulator.simulate_instance(
    instance_id,
    use_fatigue=use_fatigue,
)

# Print key outputs
print(f"Instance: {result.instance_id}, | Mode: {result.mode} | Cmax: {result.makespan}")

print("\nmetrics:")
for key, value in sorted(result.metrics.items()):
    print(f"{key}: {value}")

# Save the Gantt chart
plot_path = Path(f"{instance_id}_{result.mode}.png")
simulator.gantt_generator(result, plot_path, plot_gantt, save_gantt, use_fatigue=use_fatigue)