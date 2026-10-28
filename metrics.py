"""Once-a-second resource sampler: this process vs the whole system (and the heaviest other GPU user, i.e. the game)."""

import csv
import os
import threading
import time
from collections import defaultdict

import psutil

try:
    import pynvml
except ImportError:
    pynvml = None
try:
    import win32pdh
except ImportError:
    win32pdh = None


class GpuCounters:
    """Per-process GPU utilization and dedicated VRAM from Windows perf counters (what Task Manager shows)."""

    def __init__(self):
        self.q = win32pdh.OpenQuery()
        self.util = win32pdh.AddEnglishCounter(self.q, r"\GPU Engine(*)\Utilization Percentage")
        self.mem = win32pdh.AddEnglishCounter(self.q, r"\GPU Process Memory(*)\Dedicated Usage")
        win32pdh.CollectQueryData(self.q)

    @staticmethod
    def _pid(instance: str) -> int:
        return int(instance.split("_")[1])

    def sample(self) -> tuple[dict[int, float], dict[int, int]]:
        win32pdh.CollectQueryData(self.q)
        try:
            util_raw = win32pdh.GetFormattedCounterArray(self.util, win32pdh.PDH_FMT_DOUBLE)
        except Exception:
            util_raw = {}
        try:
            mem_raw = win32pdh.GetFormattedCounterArray(self.mem, win32pdh.PDH_FMT_LARGE)
        except Exception:
            mem_raw = {}
        # per pid: sum engines of the same type, take the busiest type (Task Manager's "GPU" column)
        by_type: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for inst, v in util_raw.items():
            by_type[self._pid(inst)][inst.rsplit("engtype_", 1)[-1]] += v
        util = {pid: min(100.0, max(types.values())) for pid, types in by_type.items()}
        mem: dict[int, int] = defaultdict(int)
        for inst, v in mem_raw.items():
            mem[self._pid(inst)] += v
        return util, dict(mem)


FIELDS = [
    "time", "tts_enabled", "busy", "queue",
    "app_cpu_pct", "app_ram_gib", "app_vram_gib", "app_gpu_pct",
    "sys_cpu_pct", "sys_ram_gib", "gpu_util_pct", "gpu_vram_used_gib", "gpu_vram_total_gib",
    "game_name", "game_vram_gib", "game_gpu_pct",
]


class MetricsRecorder:
    def __init__(self, csv_path, state_fn, interval: float = 1.0):
        self.proc = psutil.Process(os.getpid())
        self.ncpu = psutil.cpu_count()
        self.state_fn = state_fn  # -> dict(tts_enabled, busy, queue)
        self.interval = interval
        self.latest: dict = {}
        self.history: list[dict] = []
        self.gpu = GpuCounters() if win32pdh else None
        self.nvml = None
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.nvml = None
        self.file = open(csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.names: dict[int, str] = {}
        self.proc.cpu_percent()
        psutil.cpu_percent()
        threading.Thread(target=self._run, daemon=True).start()

    def _name(self, pid: int) -> str:
        if pid not in self.names:
            try:
                self.names[pid] = psutil.Process(pid).name()
            except Exception:
                self.names[pid] = f"pid {pid}"
        return self.names[pid]

    def _sample(self) -> dict:
        me = self.proc.pid
        row = {
            "time": time.strftime("%H:%M:%S"),
            **self.state_fn(),
            # cpu_percent of a process is per-core (up to 100*ncpu); normalize to the whole CPU like Task Manager
            "app_cpu_pct": round(self.proc.cpu_percent() / self.ncpu, 1),
            "app_ram_gib": round(self.proc.memory_info().rss / 2**30, 2),
            "sys_cpu_pct": psutil.cpu_percent(),
            "sys_ram_gib": round(psutil.virtual_memory().used / 2**30, 2),
        }
        if self.nvml:
            row["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(self.nvml).gpu
            mi = pynvml.nvmlDeviceGetMemoryInfo(self.nvml)
            row["gpu_vram_used_gib"] = round(mi.used / 2**30, 2)
            row["gpu_vram_total_gib"] = round(mi.total / 2**30, 2)
        if self.gpu:
            util, mem = self.gpu.sample()
        elif self.nvml:
            util, mem = self._nvml_processes()
        if self.gpu or self.nvml:
            row["app_gpu_pct"] = round(util.get(me, 0.0), 1)
            row["app_vram_gib"] = round(mem.get(me, 0) / 2**30, 2)
            others = {pid: m for pid, m in mem.items() if pid != me}
            if others:
                game = max(others, key=others.get)
                row["game_name"] = self._name(game)
                row["game_vram_gib"] = round(others[game] / 2**30, 2)
                row["game_gpu_pct"] = round(util.get(game, 0.0), 1)
        return row

    def _nvml_processes(self) -> tuple[dict[int, float], dict[int, int]]:
        """Per-process GPU load and VRAM from NVML (Linux). Inside a container, PIDs of other processes don't match."""
        mem: dict[int, int] = {}
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.nvml) +                 pynvml.nvmlDeviceGetGraphicsRunningProcesses(self.nvml)
        except pynvml.NVMLError:
            procs = []
        for p in procs:
            mem[p.pid] = mem.get(p.pid, 0) + (p.usedGpuMemory or 0)
        util: dict[int, float] = {}
        try:
            since = int((time.time() - 2 * self.interval) * 1e6)
            for s in pynvml.nvmlDeviceGetProcessUtilization(self.nvml, since):
                util[s.pid] = max(util.get(s.pid, 0.0), float(s.smUtil))
        except pynvml.NVMLError:
            pass  # not supported on every GPU/driver
        return util, mem

    def _run(self):
        while True:
            t0 = time.monotonic()
            try:
                row = self._sample()
                self.latest = row
                self.history.append(row)
                del self.history[:-600]
                self.writer.writerow(row)
                self.file.flush()
            except Exception as e:
                self.latest = {"error": str(e)}
            time.sleep(max(0.0, self.interval - (time.monotonic() - t0)))
