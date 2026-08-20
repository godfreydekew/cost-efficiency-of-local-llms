import gc
import subprocess
import threading
import time
import torch
from transformers import TrainerCallback


class PowerMonitor:
    """
    Runs in a background thread during training or inference.
    Samples GPU power at specified intervals (default 5s for training, 0.2s for inference).
    Call start() before training/inference, stop() after.
    """
    def __init__(self, interval_s=5):
        self.interval = interval_s
        self.readings = []  # all power samples in watts
        self._running = False
        self._thread = None

    def _sample_loop(self):
        while self._running:
            try:
                result = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=power.draw',
                        '--format=csv,noheader,nounits'
                    ],
                    capture_output=True, text=True, timeout=3
                )
                watts = float(result.stdout.strip().split('\n')[0])
                self.readings.append(watts)
            except Exception:
                pass  # skip failed samples silently
            time.sleep(self.interval)

    def start(self):
        self.readings = []
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True  # thread dies automatically if main process exits
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def summary(self, duration_s):
        """Returns energy stats for a completed execution period."""
        if not self.readings:
            return {
                'n_samples': 0,
                'avg_power_w': 0.0,
                'peak_power_w': 0.0,
                'energy_wh': 0.0
            }
        avg_w = sum(self.readings) / len(self.readings)
        peak_w = max(self.readings)
        # energy (Wh) = average power (W) * time (seconds) / 3600
        energy_wh = (avg_w * duration_s) / 3600
        return {
            'n_samples': len(self.readings),
            'avg_power_w': round(avg_w, 1),
            'peak_power_w': round(peak_w, 1),
            'energy_wh': round(energy_wh, 4),
        }


def reset_vram():
    """Resets PyTorch peak GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb():
    """Returns peak allocated PyTorch GPU memory in megabytes."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def clear_gpu_memory():
    """Frees unreferenced GPU memory and clears cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class EpochLogger(TrainerCallback):
    """
    Hugging Face Trainer Callback.
    Logs time, peak VRAM, accuracy, macro-F1, and power/energy per epoch.
    """
    def __init__(self, power_monitor: PowerMonitor, epoch_logs_list: list):
        self.power_monitor = power_monitor
        self.epoch_logs = epoch_logs_list
        self.epoch_start = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        reset_vram()
        self.epoch_start = time.time()
        self.power_monitor.start()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.power_monitor.stop()
            epoch_time = time.time() - (self.epoch_start or time.time())
            vram = peak_vram_mb()
            power_stats = self.power_monitor.summary(epoch_time)

            acc = metrics.get('eval_accuracy', metrics.get('accuracy', 0))
            f1 = metrics.get('eval_macro_f1', metrics.get('macro_f1', 0))

            log = {
                'epoch': int(state.epoch),
                'epoch_time_s': round(epoch_time, 2),
                'peak_vram_mb': round(vram, 1),
                'eval_accuracy': round(acc, 4),
                'eval_macro_f1': round(f1, 4),
                **power_stats,
            }
            self.epoch_logs.append(log)
            print(
                f"  Epoch {log['epoch']:>2} | "
                f"time: {epoch_time:>6.1f}s | "
                f"VRAM: {vram:>7.1f} MB | "
                f"macro F1: {log['eval_macro_f1']:.4f} | "
                f"acc: {log['eval_accuracy']:.4f} | "
                f"Power: {power_stats['avg_power_w']:.0f}W | "
                f"Energy: {power_stats['energy_wh']:.3f}Wh"
            )
