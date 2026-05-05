import argparse
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
except ImportError:
    pynvml = None


@dataclass
class GpuSnapshot:
    util_percent: float
    mem_used_mib: float
    mem_total_mib: float
    temp_c: float
    power_w: float


class GpuProbe:
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.mode = None
        self.handle = None
        self.name = "Unknown"

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(self.name, bytes):
                    self.name = self.name.decode("utf-8", errors="ignore")
                self.mode = "nvml"
            except Exception:
                self.mode = None

        if self.mode is None:
            nvsmi_ok = self._can_use_nvidia_smi()
            if nvsmi_ok:
                self.mode = "nvidia-smi"

    def _can_use_nvidia_smi(self) -> bool:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                "-i",
                str(self.gpu_index),
            ]
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=3)
            parts = [x.strip() for x in output.split(",")]
            if len(parts) >= 6:
                self.name = parts[0]
                return True
        except Exception:
            return False
        return False

    def available(self) -> bool:
        return self.mode is not None

    def read(self) -> Optional[GpuSnapshot]:
        if self.mode == "nvml":
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                try:
                    temp = float(
                        pynvml.nvmlDeviceGetTemperature(
                            self.handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                except Exception:
                    temp = -1.0
                try:
                    power = float(pynvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
                except Exception:
                    power = -1.0
                return GpuSnapshot(
                    util_percent=float(util.gpu),
                    mem_used_mib=float(mem.used) / (1024 * 1024),
                    mem_total_mib=float(mem.total) / (1024 * 1024),
                    temp_c=temp,
                    power_w=power,
                )
            except Exception:
                return None

        if self.mode == "nvidia-smi":
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_index),
                ]
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=3)
                parts = [x.strip() for x in output.split(",")]
                if len(parts) < 6:
                    return None
                self.name = parts[0]
                util = float(parts[1])
                mem_used = float(parts[2])
                mem_total = float(parts[3])
                temp = float(parts[4])
                power = float(parts[5]) if parts[5] not in {"N/A", "[Not Supported]"} else -1.0
                return GpuSnapshot(
                    util_percent=util,
                    mem_used_mib=mem_used,
                    mem_total_mib=mem_total,
                    temp_c=temp,
                    power_w=power,
                )
            except Exception:
                return None

        return None

    def close(self) -> None:
        if self.mode == "nvml":
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def safe_cpu_percent() -> float:
    if psutil is None:
        return -1.0
    try:
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return -1.0


def safe_ram_percent() -> float:
    if psutil is None:
        return -1.0
    try:
        return float(psutil.virtual_memory().percent)
    except Exception:
        return -1.0


def benchmark_torch(device: str, size: int, steps: int) -> Optional[float]:
    if torch is None:
        return None

    if device == "cuda" and (not torch.cuda.is_available()):
        return None

    try:
        a = torch.randn((size, size), device=device)
        b = torch.randn((size, size), device=device)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(steps):
            c = a @ b
            _ = c.mean()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return elapsed / steps
    except Exception:
        return None


def run_probe(duration_sec: int, interval_sec: float, gpu_index: int) -> None:
    probe = GpuProbe(gpu_index=gpu_index)

    cuda_ready = bool(torch is not None and torch.cuda.is_available())
    gpu_snapshot = probe.read() if probe.available() else None
    gpu_name = probe.name if probe.available() else "No NVIDIA GPU metrics source"

    print("=" * 72)
    print("GPU Capability Check")
    print("=" * 72)
    print(f"Torch installed: {torch is not None}")
    print(f"Torch CUDA available: {cuda_ready}")
    if torch is not None and cuda_ready:
        print(f"Torch CUDA device: {torch.cuda.get_device_name(gpu_index)}")
    print(f"GPU metrics source: {probe.mode if probe.mode else 'unavailable'}")
    print(f"Detected GPU: {gpu_name}")

    if gpu_snapshot is None:
        print("Warning: Live GPU metrics unavailable. Install pynvml or ensure nvidia-smi is in PATH.")

    timestamps = []
    gpu_util = []
    gpu_mem_percent = []
    cpu_util = []
    ram_util = []

    plt.ion()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("System Resource Monitor (Live)")

    start = time.time()
    while True:
        now = time.time()
        elapsed = now - start
        if elapsed > duration_sec:
            break

        g = probe.read() if probe.available() else None
        c = safe_cpu_percent()
        r = safe_ram_percent()

        timestamps.append(elapsed)
        cpu_util.append(c)
        ram_util.append(r)

        if g is not None and g.mem_total_mib > 0:
            gpu_util.append(g.util_percent)
            gpu_mem_percent.append((g.mem_used_mib / g.mem_total_mib) * 100.0)
            left_title = (
                f"GPU: {probe.name}\n"
                f"Temp: {g.temp_c:.1f} C | Power: {g.power_w:.1f} W | "
                f"VRAM: {g.mem_used_mib:.0f}/{g.mem_total_mib:.0f} MiB"
            )
        else:
            gpu_util.append(0.0)
            gpu_mem_percent.append(0.0)
            left_title = "GPU metrics unavailable"

        ax_left.clear()
        ax_left.plot(timestamps, gpu_util, label="GPU Util %")
        ax_left.plot(timestamps, gpu_mem_percent, label="GPU VRAM %")
        ax_left.set_title(left_title)
        ax_left.set_xlabel("Time (s)")
        ax_left.set_ylabel("Percent")
        ax_left.set_ylim(0, 100)
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(loc="upper right")

        ax_right.clear()
        ax_right.plot(timestamps, cpu_util, label="CPU Util %")
        ax_right.plot(timestamps, ram_util, label="RAM Util %")
        ax_right.set_title("CPU / RAM")
        ax_right.set_xlabel("Time (s)")
        ax_right.set_ylabel("Percent")
        ax_right.set_ylim(0, 100)
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(loc="upper right")

        fig.tight_layout()
        plt.pause(0.001)
        time.sleep(interval_sec)

    print("\nRunning compute benchmark (lower is better; seconds per matmul step)...")

    cpu_time = benchmark_torch(device="cpu", size=1024, steps=20) if torch is not None else None
    gpu_time = benchmark_torch(device="cuda", size=1024, steps=50) if cuda_ready else None

    print(f"CPU benchmark: {cpu_time if cpu_time is not None else 'N/A'}")
    print(f"GPU benchmark: {gpu_time if gpu_time is not None else 'N/A'}")

    if cpu_time is not None and gpu_time is not None and gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"GPU speedup vs CPU: {speedup:.2f}x")
        gpu_ok = speedup > 1.2
    else:
        speedup = None
        gpu_ok = cuda_ready

    fig2, ax = plt.subplots(1, 1, figsize=(8, 4))
    labels = []
    values = []

    if cpu_time is not None:
        labels.append("CPU")
        values.append(cpu_time)
    if gpu_time is not None:
        labels.append("GPU")
        values.append(gpu_time)

    if values:
        ax.bar(labels, values)
        ax.set_ylabel("Seconds / Step")
        ax.set_title("Compute Benchmark")

    if speedup is not None:
        summary = f"CUDA Ready: {cuda_ready} | Speedup: {speedup:.2f}x"
    else:
        summary = f"CUDA Ready: {cuda_ready} | Speedup: N/A"

    plt.figtext(0.5, 0.01, summary, ha="center", fontsize=10)
    fig2.tight_layout(rect=[0, 0.03, 1, 1])

    print("\nFinal verdict:")
    if gpu_ok and cuda_ready:
        print("PASS: Your system can run training on NVIDIA GPU.")
    elif cuda_ready:
        print("PARTIAL: CUDA is available, but benchmark speedup was low. Check background GPU load.")
    else:
        print("FAIL: CUDA is not available in current PyTorch environment (CPU-only build likely).")

    probe.close()
    plt.ioff()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live GPU resource monitor + CUDA capability test")
    parser.add_argument("--duration", type=int, default=30, help="Monitoring duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index to monitor")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_probe(duration_sec=args.duration, interval_sec=args.interval, gpu_index=args.gpu_index)
