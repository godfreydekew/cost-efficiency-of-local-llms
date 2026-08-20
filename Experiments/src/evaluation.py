import time
import numpy as np
from src.monitoring import PowerMonitor, peak_vram_mb, reset_vram
from src.metrics import evaluate_predictions


def run_inference_benchmark(
    trainer,
    test_dataset,
    label_names_dict,
    model_name: str,
    sample_interval_s: float = 0.2
):
    """
    Runs a single inference pass over the test dataset with GPU power sampling
    via PowerMonitor. Throughput, latency (ms/sample), peak VRAM, energy
    (Wh, energy/1k queries), and classification metrics are all computed
    directly from this one real pass over the held-out test set.
    """
    n_samples = len(test_dataset)

    print(f"\nRunning inference on test set ({n_samples:,} samples)...")
    reset_vram()

    # High-frequency power monitoring for inference
    inf_monitor = PowerMonitor(interval_s=sample_interval_s)
    inf_monitor.start()
    inf_start = time.time()

    predictions = trainer.predict(test_dataset)

    inf_time = time.time() - inf_start
    inf_monitor.stop()
    vram_inf = peak_vram_mb()

    # Process metrics
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    perf_metrics = evaluate_predictions(labels, preds, label_names_dict)

    # Power & Energy summary
    power_stats = inf_monitor.summary(inf_time)

    # Inference throughput and latency
    ms_per_sample = (inf_time / n_samples) * 1000.0
    samples_per_sec = n_samples / inf_time
    energy_wh_per_1k = (power_stats['energy_wh'] / n_samples) * 1000.0 if n_samples > 0 else 0.0
    energy_mj_per_sample = (power_stats['energy_wh'] * 3600 * 1000 / n_samples) if n_samples > 0 else 0.0

    summary = {
        'total_samples': n_samples,
        'inference_time_total_s': round(inf_time, 2),
        'samples_per_sec': round(samples_per_sec, 1),
        'ms_per_sample': round(ms_per_sample, 3),
        'avg_power_w': power_stats['avg_power_w'],
        'peak_power_w': power_stats['peak_power_w'],
        'energy_wh': power_stats['energy_wh'],
        'energy_wh_per_1k_queries': round(energy_wh_per_1k, 6),
        'energy_mj_per_sample': round(energy_mj_per_sample, 3),
        'peak_vram_mb': round(vram_inf, 1),
        'accuracy': perf_metrics['accuracy'],
        'macro_f1': perf_metrics['macro_f1'],
        'per_class_f1': perf_metrics['per_class_f1']
    }

    print(f"\n{'='*55}")
    print(f"INFERENCE BENCHMARK RESULTS — {model_name}")
    print(f"{'='*55}")
    print(f"  Total Samples Tested  : {n_samples:,}")
    print(f"  Total Inference Time  : {inf_time:.2f} s")
    print(f"  Throughput            : {samples_per_sec:.1f} samples/sec")
    print(f"  Latency               : {ms_per_sample:.3f} ms/sample")
    print(f"  Avg GPU Power         : {power_stats['avg_power_w']} W")
    print(f"  Peak GPU Power        : {power_stats['peak_power_w']} W")
    print(f"  Total Energy          : {power_stats['energy_wh']} Wh")
    print(f"  Energy / 1k Queries   : {energy_wh_per_1k:.6f} Wh")
    print(f"  Energy / Sample       : {energy_mj_per_sample:.3f} mJ/sample")
    print(f"  Peak VRAM (Inference) : {vram_inf:.1f} MB")
    print(f"{'='*55}\n")

    return summary, preds, labels
