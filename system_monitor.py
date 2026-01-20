# System Resource Monitor
# =======================
# Monitors CPU, RAM, Disk I/O, and GPU usage during operations.

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import psutil


@dataclass
class SystemMetrics:
    """Container for system resource metrics."""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_read_mb: float
    disk_write_mb: float
    timestamp: float


@dataclass
class BaselineMetrics:
    """Baseline system state captured before operations."""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    total_ram_gb: float
    cpu_count: int
    gpu_info: Optional[str] = None


def capture_baseline(sample_duration: float = 1.0) -> BaselineMetrics:
    """Capture baseline system metrics before starting work.

    Args:
        sample_duration: Seconds to sample CPU usage for accurate reading.

    Returns:
        BaselineMetrics with current system state.
    """
    # Sample CPU over the duration for accurate reading
    cpu_percent = psutil.cpu_percent(interval=sample_duration)

    memory = psutil.virtual_memory()

    # Try to get GPU info
    gpu_info = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip()
    except Exception:
        pass

    return BaselineMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_used_mb=memory.used / (1024 * 1024),
        memory_available_mb=memory.available / (1024 * 1024),
        total_ram_gb=memory.total / (1024 * 1024 * 1024),
        cpu_count=psutil.cpu_count(),
        gpu_info=gpu_info,
    )


@dataclass
class MetricsCollector:
    """Collects system metrics in the background.

    Usage:
        baseline = capture_baseline()
        collector = MetricsCollector()
        collector.start()
        # ... do work ...
        collector.stop()
        summary = collector.get_summary()
        print_comparison(baseline, summary)
    """

    samples: List[SystemMetrics] = field(default_factory=list)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    _initial_disk_read: float = 0.0
    _initial_disk_write: float = 0.0

    def _get_disk_io(self):
        """Get disk I/O counters safely."""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                return disk_io.read_bytes, disk_io.write_bytes
        except Exception:
            pass
        return 0, 0

    def _collect_metrics(self):
        """Background thread function to collect metrics."""
        while not self._stop_event.is_set():
            try:
                # Get CPU percentage (non-blocking)
                cpu_percent = psutil.cpu_percent(interval=None)

                # Get memory info
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_mb = memory.used / (1024 * 1024)

                # Get disk I/O (cumulative since start)
                disk_read, disk_write = self._get_disk_io()
                disk_read_mb = (disk_read - self._initial_disk_read) / (1024 * 1024)
                disk_write_mb = (disk_write - self._initial_disk_write) / (1024 * 1024)

                sample = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_mb=memory_used_mb,
                    disk_read_mb=disk_read_mb,
                    disk_write_mb=disk_write_mb,
                    timestamp=time.time(),
                )
                self.samples.append(sample)
            except Exception:
                pass  # Skip sample on error

            time.sleep(0.5)  # Sample every 500ms

    def start(self):
        """Start collecting metrics."""
        self._stop_event.clear()
        self.samples.clear()

        # Initialize CPU percent measurement
        psutil.cpu_percent(interval=None)

        # Store initial disk I/O counters
        self._initial_disk_read, self._initial_disk_write = self._get_disk_io()

        self._thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop collecting metrics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_summary(self) -> dict:
        """Calculate summary statistics from collected samples."""
        if not self.samples:
            return {}

        cpu_values = [s.cpu_percent for s in self.samples]
        mem_values = [s.memory_percent for s in self.samples]
        mem_used_values = [s.memory_used_mb for s in self.samples]

        # Get final disk I/O values
        last_sample = self.samples[-1]

        return {
            "duration_seconds": self.samples[-1].timestamp - self.samples[0].timestamp,
            "samples_count": len(self.samples),
            "cpu": {
                "avg_percent": sum(cpu_values) / len(cpu_values),
                "max_percent": max(cpu_values),
                "min_percent": min(cpu_values),
            },
            "memory": {
                "avg_percent": sum(mem_values) / len(mem_values),
                "max_percent": max(mem_values),
                "peak_used_mb": max(mem_used_values),
            },
            "disk_io": {
                "total_read_mb": last_sample.disk_read_mb,
                "total_write_mb": last_sample.disk_write_mb,
            },
        }


def _format_diff(
    before: float, after: float, unit: str = "%", higher_is_bad: bool = True
) -> str:
    """Format a difference with color indicator."""
    diff = after - before
    if abs(diff) < 0.1:
        indicator = "━"  # No change
    elif (diff > 0 and higher_is_bad) or (diff < 0 and not higher_is_bad):
        indicator = "▲"  # Increased (usually bad for resources)
    else:
        indicator = "▼"  # Decreased

    sign = "+" if diff > 0 else ""
    return f"{after:.1f}{unit} ({sign}{diff:.1f}{unit} {indicator})"


def print_comparison(baseline: BaselineMetrics, summary: dict):
    """Print comparison between baseline and during-test metrics."""
    print("\n" + "=" * 70)
    print("SYSTEM RESOURCE COMPARISON: BEFORE vs DURING INSPECTION")
    print("=" * 70)

    # System Info Header
    print(
        f"\n📌 System: {baseline.cpu_count} CPU cores | {baseline.total_ram_gb:.1f} GB RAM"
    )

    if not summary:
        print("\n⚠️  No metrics collected during inspection")
        print(f"\n🔹 Baseline CPU: {baseline.cpu_percent:.1f}%")
        print(f"🔹 Baseline Memory: {baseline.memory_percent:.1f}%")
        return

    duration = summary.get("duration_seconds", 0)
    print(
        f"📊 Monitoring Duration: {duration:.2f}s ({summary.get('samples_count', 0)} samples)"
    )

    # CPU Comparison
    cpu = summary.get("cpu", {})
    cpu_avg = cpu.get("avg_percent", 0)
    cpu_max = cpu.get("max_percent", 0)

    print("\n" + "-" * 70)
    print("🖥️  CPU USAGE")
    print("-" * 70)
    print(f"   {'Metric':<20} {'Before':<15} {'During (Avg)':<20} {'During (Max)':<15}")
    print(f"   {'-' * 20} {'-' * 15} {'-' * 20} {'-' * 15}")
    print(
        f"   {'CPU Utilization':<20} {baseline.cpu_percent:>6.1f}%        "
        f"{_format_diff(baseline.cpu_percent, cpu_avg):<20} {cpu_max:>6.1f}%"
    )

    # Memory Comparison
    mem = summary.get("memory", {})
    mem_avg = mem.get("avg_percent", 0)
    mem_max = mem.get("max_percent", 0)
    mem_peak_mb = mem.get("peak_used_mb", 0)

    print("\n" + "-" * 70)
    print("💾 MEMORY (RAM) USAGE")
    print("-" * 70)
    print(f"   {'Metric':<20} {'Before':<15} {'During (Avg)':<20} {'During (Max)':<15}")
    print(f"   {'-' * 20} {'-' * 15} {'-' * 20} {'-' * 15}")
    print(
        f"   {'Memory %':<20} {baseline.memory_percent:>6.1f}%        "
        f"{_format_diff(baseline.memory_percent, mem_avg):<20} {mem_max:>6.1f}%"
    )
    print(
        f"   {'Memory Used':<20} {baseline.memory_used_mb:>6.0f} MB      "
        f"{_format_diff(baseline.memory_used_mb, mem_peak_mb, ' MB'):<20}"
    )

    # Memory delta
    mem_delta = mem_peak_mb - baseline.memory_used_mb
    print(f"\n   💡 Additional memory used during inspection: {mem_delta:+.1f} MB")

    # Disk I/O
    disk = summary.get("disk_io", {})
    print("\n" + "-" * 70)
    print("💿 DISK I/O (during inspection only)")
    print("-" * 70)
    print(f"   • Total Read:  {disk.get('total_read_mb', 0):.2f} MB")
    print(f"   • Total Write: {disk.get('total_write_mb', 0):.2f} MB")

    # GPU Comparison
    if baseline.gpu_info:
        print("\n" + "-" * 70)
        print("🎮 GPU INFO (NVIDIA) - Baseline")
        print("-" * 70)
        for line in baseline.gpu_info.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                name, mem_used, mem_total, util = parts[:4]
                print(f"   • {name}: {mem_used}/{mem_total} MB | Utilization: {util}%")

    # Summary
    print("\n" + "=" * 70)
    print("📈 IMPACT SUMMARY")
    print("=" * 70)
    cpu_impact = cpu_avg - baseline.cpu_percent
    mem_impact = mem_max - baseline.memory_percent

    if cpu_impact > 50:
        cpu_status = "🔴 HIGH"
    elif cpu_impact > 20:
        cpu_status = "🟡 MODERATE"
    else:
        cpu_status = "🟢 LOW"

    if mem_impact > 20:
        mem_status = "🔴 HIGH"
    elif mem_impact > 10:
        mem_status = "🟡 MODERATE"
    else:
        mem_status = "🟢 LOW"

    print(f"   CPU Impact:    {cpu_status} ({cpu_impact:+.1f}% from baseline)")
    print(f"   Memory Impact: {mem_status} ({mem_impact:+.1f}% from baseline)")


def print_system_metrics(summary: dict):
    """Print formatted system metrics summary (without comparison)."""
    if not summary:
        print("⚠️  No system metrics collected")
        return

    print("\n" + "=" * 60)
    print("SYSTEM RESOURCE USAGE DURING INSPECTION")
    print("=" * 60)

    duration = summary.get("duration_seconds", 0)
    print(f"\n📊 Monitoring Duration: {duration:.2f} seconds")
    print(f"   Samples Collected: {summary.get('samples_count', 0)}")

    # CPU metrics
    cpu = summary.get("cpu", {})
    print("\n🖥️  CPU Usage:")
    print(f"   • Average: {cpu.get('avg_percent', 0):.1f}%")
    print(f"   • Maximum: {cpu.get('max_percent', 0):.1f}%")
    print(f"   • Minimum: {cpu.get('min_percent', 0):.1f}%")

    # Memory metrics
    mem = summary.get("memory", {})
    print("\n💾 Memory (RAM) Usage:")
    print(f"   • Average: {mem.get('avg_percent', 0):.1f}%")
    print(f"   • Maximum: {mem.get('max_percent', 0):.1f}%")
    print(f"   • Peak Used: {mem.get('peak_used_mb', 0):.1f} MB")

    # Disk I/O metrics
    disk = summary.get("disk_io", {})
    print("\n💿 Disk I/O (during inspection):")
    print(f"   • Total Read: {disk.get('total_read_mb', 0):.2f} MB")
    print(f"   • Total Write: {disk.get('total_write_mb', 0):.2f} MB")

    # System info
    print("\n📌 Current System State:")
    total_mem = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    cpu_count = psutil.cpu_count()
    print(f"   • Total RAM: {total_mem:.1f} GB")
    print(f"   • CPU Cores: {cpu_count}")

    # Try to get GPU info if available
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("\n🎮 GPU Info (NVIDIA):")
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    name, mem_used, mem_total, util = parts[:4]
                    print(f"   • {name}")
                    print(f"     Memory: {mem_used} / {mem_total} MB")
                    print(f"     Utilization: {util}%")
    except Exception:
        pass  # nvidia-smi not available or error
