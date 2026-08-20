"""
Evaluation Engine for Zero-Shot and Few-Shot LLM Classification Benchmarks.
Dissertation: Cost/Efficiency of Deploying Local LLMs for Text Classification
"""

import os
import time
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from src.config import RESULTS_JSON_DIR
from src.metrics import evaluate_predictions
from src.monitoring import PowerMonitor, peak_vram_mb, reset_vram
from src.prompting import (
    build_system_prompt,
    build_user_message,
    parse_prediction,
    classify_gpt,
    classify_llama
)


def run_prompt_benchmark(
    backend: str,  # 'llama' or 'gpt'
    test_dataset,
    label_names_dict: Dict[int, str],
    model_name: str,
    examples: Optional[List[Dict[str, Any]]] = None,
    model = None,
    tokenizer = None,
    client = None,
    text_col: str = "text",
    label_col: str = "label",
    sample_interval_s: float = 0.2,
    save_results: bool = True,
    experiment_name: str = "prompt_experiment",
) -> Tuple[Dict[str, Any], List[int], List[int]]:
    """
    Runs a complete evaluation benchmark over the held-out test set for Zero-Shot or Few-Shot prompting.
    
    Tracks and logs:
    1. Inference Timing & Throughput: total_time_s, samples_per_sec, ms_per_sample
    2. Instruction Compliance / Parsing: primary_parse_rate, fallback_parse_rate, failed_parse_rate
    3. Token & Cost Usage: input_tokens, output_tokens, total_tokens, actual_cost_usd
    4. Hardware & Power (for local Llama): peak_vram_mb, avg_power_w, energy_wh, energy_mj_per_sample
    5. Classification Metrics: accuracy, macro_f1, per_class_f1
    
    Returns:
        Tuple of (summary_results_dict, predictions_list, ground_truth_labels_list)
    """
    n_samples = len(test_dataset)
    valid_labels = set(label_names_dict.keys())
    mode_str = "Few-Shot" if examples else "Zero-Shot"
    
    print(f"\n{'='*65}")
    print(f"PROMPT BENCHMARK RUN: {model_name} ({mode_str})")
    print(f"Backend     : {backend.upper()}")
    print(f"Test Set    : {n_samples:,} samples")
    print(f"{'='*65}\n")
    
    predictions = []
    labels = []
    
    # Token and Cost Accumulators
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost_usd = 0.0
    
    # Parser Compliance Accumulators
    primary_parse_count = 0
    fallback_parse_count = 0
    failed_parse_count = 0
    
    # Power and Memory Monitoring for Local Runs
    power_monitor = None
    if backend == 'llama':
        reset_vram()
        power_monitor = PowerMonitor(interval_s=sample_interval_s)
        power_monitor.start()
        
    start_time = time.time()
    
    # Benchmark Loop
    for sample in tqdm(test_dataset, desc=f"Evaluating {mode_str}"):
        text = sample[text_col]
        true_label = sample[label_col]
        labels.append(true_label)
        
        if backend == 'gpt':
            raw_output, in_tok, out_tok, tot_tok, cost = classify_gpt(
                text=text,
                label_names=label_names_dict,
                examples=examples,
                client=client,
                model=model_name
            )
            total_cost_usd += cost
        elif backend == 'llama':
            raw_output, in_tok, out_tok, tot_tok, _ = classify_llama(
                text=text,
                label_names=label_names_dict,
                examples=examples,
                model=model,
                tokenizer=tokenizer
            )
        else:
            raise ValueError(f"Unsupported backend '{backend}'. Must be 'llama' or 'gpt'.")
            
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        total_tokens += tot_tok
        
        # Parse model prediction
        pred, parse_status = parse_prediction(raw_output, valid_labels)
        predictions.append(pred)
        
        if parse_status:
            primary_parse_count += 1
        elif pred != -1:
            fallback_parse_count += 1
        else:
            failed_parse_count += 1
            
    total_inference_time_s = time.time() - start_time
    
    # Stop Power Monitoring if applicable
    vram_peak = 0.0
    power_stats = {'avg_power_w': 0.0, 'peak_power_w': 0.0, 'energy_wh': 0.0}
    if backend == 'llama':
        if power_monitor:
            power_monitor.stop()
            power_stats = power_monitor.summary(total_inference_time_s)
        vram_peak = peak_vram_mb()
        
    # Calculate Latency & Throughput Metrics
    ms_per_sample = (total_inference_time_s / n_samples) * 1000.0 if n_samples > 0 else 0.0
    samples_per_sec = n_samples / total_inference_time_s if total_inference_time_s > 0 else 0.0
    energy_wh_per_1k = (power_stats['energy_wh'] / n_samples) * 1000.0 if n_samples > 0 else 0.0
    energy_mj_per_sample = (power_stats['energy_wh'] * 3600 * 1000 / n_samples) if n_samples > 0 else 0.0
    
    # Calculate Parser Rates
    primary_parse_rate = primary_parse_count / n_samples if n_samples > 0 else 0.0
    fallback_parse_rate = fallback_parse_count / n_samples if n_samples > 0 else 0.0
    failed_parse_rate = failed_parse_count / n_samples if n_samples > 0 else 0.0
    
    # Calculate Classification Performance Metrics
    perf_metrics = evaluate_predictions(labels, predictions, label_names_dict)
    
    # Construct Summary Results Schema
    summary = {
        'model_name': model_name,
        'backend': backend,
        'mode': mode_str,
        'num_few_shot_examples': len(examples) if examples else 0,
        'total_samples': n_samples,
        
        # Timing & Throughput
        'inference_time_total_s': round(total_inference_time_s, 2),
        'samples_per_sec': round(samples_per_sec, 2),
        'ms_per_sample': round(ms_per_sample, 3),
        
        # Token Logging & API Cost
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_tokens,
        'actual_cost_usd': round(total_cost_usd, 6),
        'avg_input_tokens_per_sample': round(total_input_tokens / n_samples, 1) if n_samples > 0 else 0,
        'avg_output_tokens_per_sample': round(total_output_tokens / n_samples, 2) if n_samples > 0 else 0,
        
        # Instruction Compliance / Parser
        'primary_parse_rate': round(primary_parse_rate, 4),
        'fallback_parse_rate': round(fallback_parse_rate, 4),
        'failed_parse_rate': round(failed_parse_rate, 4),
        'primary_parse_count': primary_parse_count,
        'fallback_parse_count': fallback_parse_count,
        'failed_parse_count': failed_parse_count,
        
        # GPU Power & Energy (Local Runs)
        'peak_vram_mb': round(vram_peak, 1),
        'avg_power_w': power_stats['avg_power_w'],
        'peak_power_w': power_stats['peak_power_w'],
        'energy_wh': power_stats['energy_wh'],
        'energy_wh_per_1k_queries': round(energy_wh_per_1k, 6),
        'energy_mj_per_sample': round(energy_mj_per_sample, 3),
        
        # Performance Classification Metrics
        'accuracy': perf_metrics['accuracy'],
        'macro_f1': perf_metrics['macro_f1'],
        'per_class_f1': perf_metrics['per_class_f1']
    }
    
    # Print Summary to Terminal / Notebook Output
    print(f"\n{'='*65}")
    print(f"BENCHMARK RESULTS — {model_name} ({mode_str})")
    print(f"{'='*65}")
    print(f"  Total Samples Tested     : {n_samples:,}")
    print(f"  Total Inference Time     : {total_inference_time_s:.2f} s")
    print(f"  Throughput               : {samples_per_sec:.1f} samples/sec")
    print(f"  Latency per Query        : {ms_per_sample:.3f} ms/sample")
    print(f"  Input Tokens             : {total_input_tokens:,}")
    print(f"  Output Tokens            : {total_output_tokens:,}")
    print(f"  Total Cost               : ${total_cost_usd:.4f} USD")
    print(f"  Primary Parse Rate       : {primary_parse_rate*100:.1f}%")
    print(f"  Fallback Parse Rate      : {fallback_parse_rate*100:.1f}%")
    print(f"  Accuracy                 : {perf_metrics['accuracy']:.4f}")
    print(f"  Macro F1                 : {perf_metrics['macro_f1']:.4f}")
    if backend == 'llama':
        print(f"  Avg GPU Power            : {power_stats['avg_power_w']} W")
        print(f"  Peak VRAM                : {vram_peak:.1f} MB")
    print(f"{'='*65}\n")
    
    # Save Results JSON
    if save_results:
        os.makedirs(RESULTS_JSON_DIR, exist_ok=True)
        clean_model_id = model_name.replace("/", "_").replace("-", "_")
        filename = f"{experiment_name}_{clean_model_id}_{mode_str.lower()}.json"
        filepath = Path(RESULTS_JSON_DIR) / filename
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved benchmark results to: {filepath}\n")
        
    return summary, predictions, labels
