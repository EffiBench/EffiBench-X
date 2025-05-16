import os
import json
import shutil
import sys
import signal
import hashlib
import re
import statistics
from copy import deepcopy
from pathlib import Path
from ruamel.yaml import YAML
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional, Dict, Any

import click
from tqdm import tqdm

from effibench.utils import (
    EFFIBENCH_REGISTRY, EFFIBENCH_LANGS, load_json_file, save_json_file, create_logger,
    sort_problem_files,
)
from effibench.run_tests import run_tests, postprocess_text


EFFIBENCH_DEBUG = os.environ.get("EFFIBENCH_DEBUG", "0") == "1"
TABLE_FORMAT: str = "simple"
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {}


@click.group()
def cli():
    """CLI for evaluating LLM-generated solutions."""
    pass


def hash_code_name(code: str) -> str:
    """Generate a hash for solution code for deduplication"""
    return hashlib.md5(code.encode('utf-8')).hexdigest()[:16]


def evaluate_one_solution(problem_file: Path, output_file: Path, model: str, lang: str, solution_code: str, problem_data: dict = None):
    problem_name = problem_file.stem
    if problem_data is None:
        problem_data = load_json_file(problem_file)
    log = create_logger(model, problem_name, lang)
    
    is_canonical = model.startswith("canonical_")
    if is_canonical:
        cache_dir = output_file.parent.parent / "cache"
        cache_file = cache_dir / f"{hash_code_name(solution_code)}.json"
        if cache_file.exists():
            shutil.copy(cache_file, output_file)
            log(f"Skipping evaluation for {problem_name}_{lang} (canonical solution already evaluated)", fg="yellow")
            return
    
    # Load test cases
    test_cases = problem_data["generated_tests"]
    if isinstance(test_cases, str):
        test_cases = json.loads(test_cases)
    
    try:
        results = run_tests(
            lang=lang,
            solution=solution_code,
            test_cases=test_cases,
            evaluator=problem_data["evaluator"],
            test_runner=(problem_data.get("test_runners") or {}).get(lang),
            early_stop=False,
            raise_on_error=False,
            as_batch=True,
            polling_interval=30,
        )
    except Exception as e:
        # embed model, problem_name, lang in the error message
        raise type(e)(f"{model}:{problem_name}:{lang}:{e}") from e
    
    total_tests = len(results)
    num_passed = sum(1 for r in results if r["passed"])
    if num_passed == total_tests:
        log(f"Passed all {total_tests} tests", fg="green")
    else:
        log(f"Passed {num_passed}/{total_tests} tests", fg="yellow")
        if EFFIBENCH_DEBUG:
            error_results = [r for r in results if not r["passed"]]
            for i, r in enumerate(error_results):
                log(f"Error {i+1}:exit code {r['exit_code']}:{r['text'][:500]}", fg="yellow")
    
    save_json_file(output_file, results)
    if is_canonical:
        shutil.copy(output_file, cache_file)


def _run_task(args: tuple) -> None:
    problem_file, output_file, model, lang, solution_code, problem_data = args
    evaluate_one_solution(problem_file, output_file, model, lang, solution_code, problem_data=problem_data)


def _process_chunk(chunk: list, n_threads: int) -> None:
    with ThreadPoolExecutor(max_workers=n_threads) as thread_executor:
        future_to_arg = {thread_executor.submit(_run_task, arg): arg for arg in chunk}
        for future in as_completed(future_to_arg):
            arg = future_to_arg[future]
            e = future.exception()
            if e:
                problem_file, _, model, lang, _, _ = arg
                log = create_logger(model, problem_file.stem, lang)
                log(f"{e.__class__.__name__}:{e}", fg="red")


def execute_tasks_nested(args_list: list, n_processes: int = None, n_threads: int = 1) -> None:
    if n_processes is None:
        n_processes = os.cpu_count() or 1
    chunks = [args_list[i::n_processes] for i in range(n_processes)]
    chunks = [chunk for chunk in chunks if chunk]
    with ProcessPoolExecutor(max_workers=n_processes) as process_executor:
        futures = {process_executor.submit(_process_chunk, chunk, n_threads): chunk for chunk in chunks}
        for future in as_completed(futures):
            e = future.exception()
            if e:
                print(f"Error in process chunk execution: {e}", file=sys.stderr)


@cli.command()
@click.argument("problem_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True), default=Path("data") / "dataset")
@click.argument("solution_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True), default=Path("data") / "solutions")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=Path("data") / "evaluation")
@click.option("--langs", "-l", type=str, help="Comma-separated list of languages to evaluate (e.g., 'python3,javascript'). If empty, all available languages will be evaluated.")
@click.option("--overwrite", "-f", type=bool, default=False, help="Overwrite existing evaluation results.")
@click.option("--n-processes", "-p", "n_processes", type=int, default=None, help="Number of parallel processes for evaluation (default: number of CPU cores).")
@click.option("--n-threads", "-t", "n_threads", type=int, default=1, help="Number of threads per process.")
@click.option("--model", "-m", "models", multiple=True, help="One or more model names to evaluate (can be specified multiple times).")
@click.option("--dry-run", "-d", is_flag=True, help="Only show statistics without executing evaluation.")
def evaluate(problem_dir: Path, solution_dir: Path, output_dir: Path, langs: str, overwrite: bool, n_processes: int, n_threads: int, models: tuple[str, ...], dry_run: bool) -> None:
    """Evaluate all model solutions against test cases and store results."""
    log = create_logger()
    langs = [lang.strip() for lang in langs.split(",") if lang.strip() in EFFIBENCH_LANGS] if langs else EFFIBENCH_LANGS

    problem_files = sort_problem_files(list(problem_dir.glob("*.json")))
    all_problem_data = {f.stem: load_json_file(f) for f in problem_files}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare all tasks first
    tasks = []
    tasks_by_model = defaultdict(list)

    for model_solution_file in sorted(list(solution_dir.glob("*.json"))):
        model = model_solution_file.stem
        if models and model not in models:
            continue
        log(f"Preparing tasks for model: {model}", fg="cyan")
        
        is_canonical = model.startswith("canonical_")
        if is_canonical:
            cache_dir = output_dir / "cache"
            cache_dir.mkdir(exist_ok=True, parents=True)

        model_output_dir = output_dir / model
        model_output_dir.mkdir(exist_ok=True, parents=True)
        model_solutions = load_json_file(model_solution_file)

        for problem_name, problem_solutions in model_solutions.items():
            for lang, solution_code in problem_solutions.items():
                if lang not in langs:
                    continue
                if solution_code is None:
                    continue

                problem_file = problem_dir / f"{problem_name}.json"
                output_file = model_output_dir / f"{problem_name}_{lang}.json"
                
                if not overwrite and output_file.exists():
                    continue
                
                task = {
                    "model": model,
                    "problem_name": problem_name,
                    "lang": lang,
                    "solution_code": solution_code,
                    "problem_file": problem_file,
                    "output_file": output_file,
                    "problem_data": all_problem_data[problem_name]
                }
                tasks.append(task)
                tasks_by_model[model].append(task)

    # Print statistics
    total_tasks = len(tasks)
    log(f"\nTotal evaluation tasks: {total_tasks}", fg="blue")
    log(f"{'Model':<30} {'Tasks':<10}", fg="blue")
    log("-" * 40, fg="blue")
    for model, model_tasks in sorted(tasks_by_model.items()):
        log(f"{model:<30} {len(model_tasks):<10}", fg="cyan")

    if dry_run:
        log("\nDry run completed. No evaluations performed.", fg="yellow")
        return

    if not tasks:
        log("\nNo tasks to execute. All evaluations already completed or filtered out.", fg="yellow")
        return

    log(f"\nStarting execution of {total_tasks} tasks...", fg="blue")
    
    # Execute tasks with nested process and thread pools
    task_args = [
        (task["problem_file"], task["output_file"], task["model"], task["lang"], task["solution_code"], task["problem_data"])
        for task in tasks
    ]
    try:
        execute_tasks_nested(task_args, n_processes=n_processes, n_threads=n_threads)
    except KeyboardInterrupt:
        log("Keyboard interrupt received. Exiting...", fg="yellow")
        sys.exit(1)


def is_functional(file_stem: str) -> bool:
    return file_stem.startswith("leetcode_")


def get_problem_type(file_stem: str) -> str:
    if is_functional(file_stem):
        return "functional"
    else:
        return "io"


def get_problem_source(file_stem: str) -> str:
    return file_stem.split("_")[0]


def is_passed(results: list[dict], return_stats: bool = False) -> bool:
    def is_passed_item(r: dict) -> bool:
        if r.get("passed", False):
            return True
        if r["status"] == "done" and r["text"] == "" and r["exit_code"] == 0:
            return True
        if postprocess_text(r["text"]) == postprocess_text(r["input"] + r["output"]):
            return True
        return False
    
    num_passed = sum(1 for r in results if is_passed_item(r))
    if return_stats:
        return num_passed == len(results), num_passed
    else:
        return num_passed == len(results)


def compute_model_stats(problem_names, langs, model_eval_dir, stats_file):
    """Compute evaluation statistics for a model and save to stats_file."""
    log = create_logger(model_eval_dir.name)
    if stats_file.exists():
        log(f"Found existing stats file for {model_eval_dir.name}, loading...", fg="yellow")
        return load_json_file(stats_file)
    model_stats = defaultdict(dict)
    for problem_name in tqdm(problem_names, desc=f"{model_eval_dir.name}:Processing {len(problem_names)} problems", smoothing=0.01):
        for lang in langs:
            eval_path = model_eval_dir / f"{problem_name}_{lang}.json"
            if not eval_path.exists():
                model_stats[problem_name][lang] = None
                continue
            
            try:
                eval_data = load_json_file(eval_path)
            except Exception as e:
                log(f"{e}. Error loading evaluation data for {problem_name}_{lang}, unlinking...", fg="red")
                eval_path.unlink(missing_ok=True)
                model_stats[problem_name][lang] = None
                continue
                
            eval_data = [r for r in eval_data if r["runtime"] < 10_000_000_000]
            passed, num_passed = is_passed(eval_data, return_stats=True)
            stats = {"passed": passed, "pass_rate": num_passed / len(eval_data) if eval_data else 0.0, "runtime_max": None, "runtime_sum": None, "memory": None, "integral_max": None, "integral_sum": None}
            if passed:
                stats["runtime_max"] = max(r["runtime"] / 1_000_000 for r in eval_data) # nanoseconds -> seconds
                stats["runtime_sum"] = sum(r["runtime"] / 1_000_000 for r in eval_data) # nanoseconds -> seconds
                stats["memory"] = max(r["memory"] / 1_000 for r in eval_data) # Bytes -> KB
                stats["integral_max"] = max(r["integral"] * 1_000 for r in eval_data) # microseconds -> milliseconds
                stats["integral_sum"] = sum(r["integral"] * 1_000 for r in eval_data) # microseconds -> milliseconds
            model_stats[problem_name][lang] = stats
    save_json_file(stats_file, dict(model_stats))
    log(f"Saved evaluation statistics to {stats_file}", fg="green")
    return model_stats


def compute_model_metrics(problem_names, langs, model, model_stats, canonical_runtime_stats, canonical_memory_stats):
    """
    Compute performance metrics for a model across languages and problems.
    
    Args:
        problem_names: List of problem names to analyze
        langs: List of languages to analyze
        model: Name of the model being analyzed
        model_stats: Statistics for the model being analyzed
        canonical_runtime_stats: Statistics for the canonical runtime solutions
        canonical_memory_stats: Statistics for the canonical memory solutions
        
    Returns:
        dict: Metrics data containing per-language metrics and averages
    """
    metrics_data = {}
    metrics_langs = {}
    metrics_data["per_lang"] = metrics_langs
    
    for lang in langs:
        metrics_lang = defaultdict(float)
        for problem_name in problem_names:
            if not model_stats[problem_name][lang]:
                continue
            
            metrics_lang["pass_rate"] += 1 if model_stats[problem_name][lang].get("passed", False) else 0
            
            if model in ("canonical_runtime", "canonical_memory"):
                for k in ("memory", "runtime_max", "runtime_sum", "integral_max", "integral_sum"):
                    metrics_lang[f"{k}_score"] += 1.0
                continue
            
            canonical_memory = canonical_memory_stats[problem_name][lang].get("memory", 0.0)
            model_memory = model_stats[problem_name][lang].get("memory")
            model_memory = float("inf") if model_memory is None else model_memory
            metrics_lang["memory_score"] += min(1.0, max(0.0, canonical_memory / model_memory))
            
            for k in ("runtime_max", "runtime_sum"):
                canonical_value = canonical_runtime_stats[problem_name][lang].get(k, 0.0)
                model_value = model_stats[problem_name][lang].get(k)
                model_value = float("inf") if model_value is None else model_value
                if canonical_value is None:
                    print(f"Warning: canonical {k} is None for {problem_name}_{lang}")
                metrics_lang[f"{k}_score"] += min(1.0, max(0.0, canonical_value / model_value))

            for k in ("integral_max", "integral_sum"):
                canonical_runtime_value = canonical_runtime_stats[problem_name][lang].get(k, 0.0)
                canonical_memory_value = canonical_memory_stats[problem_name][lang].get(k, 0.0)
                model_value = model_stats[problem_name][lang].get(k)
                model_value = float("inf") if model_value is None else model_value
                metrics_lang[f"{k}_score"] += min(1.0, max(0.0, min(canonical_runtime_value, canonical_memory_value) / model_value))
                
        for k in metrics_lang.keys():
            metrics_lang[k] /= len(problem_names)
        metrics_langs[lang] = dict(metrics_lang)
    
    # Calculate average across all languages
    metrics_langs_avg = {k: sum([metrics_langs[lang][k] for lang in langs]) / len(langs) for k in metrics_langs[langs[0]].keys()}
    metrics_data["avg"] = dict(metrics_langs_avg)
    
    return metrics_data


# New CLI command that merges postprocess and compare_models
@cli.command()
@click.argument("problem_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=Path("data") / "dataset")
@click.option("--evaluation-dir", "-e", "evaluation_dirs", multiple=True, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=(Path("data") / "evaluation",), help="One or more directories containing model evaluations")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.option("--langs", "-l", type=str, help="Comma-separated list of languages to process")
@click.option("--sort-by", "-s", type=str, default=None, help="Metric to sort models by")
@click.option("--table-format", "-F", type=click.Choice(["simple","latex"]), default="simple", help="Table format: simple or latex")
@click.option("--model-config", "-c", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), default=Path("model_config.yaml"), help="Path to model config YAML file")
@click.option("--all-canonical", is_flag=True, default=False)
@click.option("--model", "-m", "models", multiple=True, help="One or more model names to include in reports (can be specified multiple times)")
def report(problem_dir: Path, evaluation_dirs: tuple[Path, ...], output_dir: Path, langs: str, sort_by: str, table_format: str, model_config: Path, all_canonical: bool, models: tuple[str, ...]) -> None:
    global TABLE_FORMAT, MODEL_CONFIG
    TABLE_FORMAT = table_format
    MODEL_CONFIG = YAML().load(model_config.read_text()) if model_config.exists() else {}
    
    # Parse languages
    langs = [lang.strip() for lang in langs.split(",") if lang.strip() in EFFIBENCH_LANGS] if langs else EFFIBENCH_LANGS
    
    # Determine if we need to aggregate stats (when multiple dirs are provided)
    evaluation_dirs = list(evaluation_dirs)
    aggregate = len(evaluation_dirs) > 1
    
    # Get all problem names from dataset
    problem_files = sort_problem_files(list(problem_dir.glob("*.json")))
    problem_names = [f.stem for f in problem_files]
    
    # Process each evaluation directory and collect metrics
    metrics_maps = []
    model_stats_maps = []

    for eval_dir in evaluation_dirs:
        # Set output directory for this evaluation (for stats files)
        local_output = output_dir or eval_dir
        
        # Identify model evaluation directories, excluding cache
        excluded = {"cache"}
        model_dirs = sorted([d for d in eval_dir.glob("*") if d.is_dir() and d.name not in excluded])
        
        # Compute or load stats for each model
        model_stats_map = {}
        for model_dir in model_dirs:
            model = model_dir.name
            # Skip models not in the specified list, except canonical models
            if models and model not in models and not model.startswith("canonical_"):
                continue
            stats_file = local_output / f"stats_{model}.json"
            stats = compute_model_stats(problem_names, langs, model_dir, stats_file)
            model_stats_map[model] = stats
        
        # Prepare canonical stats intersection
        if all_canonical:
            def prune_all_langs_passed_problems(stats: dict) -> dict:
                return {p: stats[p] for p in stats if all(stats[p][lang] and stats[p][lang].get("passed", False) for lang in langs)}
            canonical_runtime_stats = prune_all_langs_passed_problems(model_stats_map.get("canonical_runtime", {}))
            canonical_memory_stats = prune_all_langs_passed_problems(model_stats_map.get("canonical_memory", {}))
            good_problems = list(set(canonical_runtime_stats) & set(canonical_memory_stats))
        else:
            canonical_models = [m for m in model_stats_map if m.startswith("canonical_")]
            good_problems = [
                p for p in problem_names
                if all(
                    any(
                        model_stats_map[c][p].get(lang) and model_stats_map[c][p][lang].get("passed", False)
                        for c in canonical_models
                    )
                    for lang in langs
                )
            ]
            canonical_runtime_stats = {}
            canonical_memory_stats = {}
            
            for p in good_problems:
                canonical_runtime_stats[p] = {}
                canonical_memory_stats[p] = {}
                
                for lang in langs:
                    best_rt_model = min(
                        (c for c in canonical_models
                         if model_stats_map[c][p].get(lang) and model_stats_map[c][p][lang].get("passed", False)),
                        key=lambda c: model_stats_map[c][p][lang]["runtime_sum"]
                    )
                    canonical_runtime_stats[p][lang] = model_stats_map[best_rt_model][p][lang]
                    
                    best_mem_model = min(
                        (c for c in canonical_models
                         if model_stats_map[c][p].get(lang) and model_stats_map[c][p][lang].get("passed", False)),
                        key=lambda c: model_stats_map[c][p][lang]["memory"]
                    )
                    canonical_memory_stats[p][lang] = model_stats_map[best_mem_model][p][lang]

        
        # Prune stats for all models to remaining problems
        for model, stats in model_stats_map.items():
            model_stats_map[model] = {k: stats[k] for k in good_problems}
        # Store model stats for later metrics computation
        model_stats_maps.append({
            "model_stats_map": model_stats_map,
            "canonical_runtime_stats": canonical_runtime_stats,
            "canonical_memory_stats": canonical_memory_stats,
            "problem_names": good_problems
        })

    if len(model_stats_maps) > 1:
        # Find intersection of problem_names across all entries
        common_problems = set(model_stats_maps[0]["problem_names"])
        for i, stats_entry in enumerate(model_stats_maps[1:]):
            common_problems &= set(stats_entry["problem_names"])
        problem_names = list(common_problems)
        
        # Filter stats based on common problems
        for stats_entry in model_stats_maps:
            stats_entry["problem_names"] = problem_names
            stats_entry["model_stats_map"] = {
                model: {p: stats[p] for p in problem_names if p in stats}
                for model, stats in stats_entry["model_stats_map"].items()
            }
            stats_entry["canonical_runtime_stats"] = {
                p: stats_entry["canonical_runtime_stats"][p]
                for p in problem_names if p in stats_entry["canonical_runtime_stats"]
            }
            stats_entry["canonical_memory_stats"] = {
                p: stats_entry["canonical_memory_stats"][p]
                for p in problem_names if p in stats_entry["canonical_memory_stats"]
            }
        
        # Synchronize
        common_models = set(model_stats_maps[0]["model_stats_map"].keys())
        for entry in model_stats_maps[1:]:
            common_models &= set(entry["model_stats_map"].keys())
        for model in common_models:
            for problem in problem_names:
                for lang in langs:
                    passed_vals = []
                    for entry in model_stats_maps:
                        stats = entry["model_stats_map"][model][problem][lang]
                        passed_vals.append(stats["passed"] if stats else False)
                    if len(set(passed_vals)) > 1:
                        for entry in model_stats_maps:
                            stats = entry["model_stats_map"][model][problem][lang]
                            if stats:
                                stats["passed"] = False

    else:
        problem_names = model_stats_maps[0]['problem_names']

    # Compute metrics maps from stats
    for stats_entry in model_stats_maps:
        m_stats_map = stats_entry["model_stats_map"]
        m_canonical_runtime_stats = stats_entry["canonical_runtime_stats"]
        m_canonical_memory_stats = stats_entry["canonical_memory_stats"]
        m_problem_names = stats_entry["problem_names"]

        metrics_map = {}
        for model, stats in m_stats_map.items():
            if model.startswith("canonical_"):
                continue
            # Skip models not in the specified list
            if models and model not in models:
                continue
            metrics_map[model] = compute_model_metrics(m_problem_names, langs, model, stats, m_canonical_runtime_stats, m_canonical_memory_stats)
        metrics_maps.append(metrics_map)

    # Merge metrics maps if aggregating
    if aggregate:
        metrics_map = merge_metrics_maps(metrics_maps)
    else:
        metrics_map = metrics_maps[0]
    
    # If there are no models left after filtering, exit early
    if not metrics_map:
        return
    
    # Define metrics for display
    simplified_metrics = {"runtime_sum_score": "runtime", "memory_score": "memory", "integral_sum_score": "integral", "pass_rate": "pass_rate"}
    # Adjust metric headers for LaTeX format
    if table_format == "latex":
        simplified_metrics = {
            "runtime_sum_score": "Execution Time (ET) (\\%)",
            "memory_score": "Memory Peak (MP) (\\%)",
            "integral_sum_score": "Memory Integral (MI) (\\%)",
            "pass_rate": "Pass@1 (\\%)"
        }
    # Print comparison tables
    print("\n===== Average scores of all models =====")
    print_filtered_model_comparison_table(metrics_map, "avg", print, sort_by, simplified_metrics)
    for lang in langs:
        print(f"\n===== Scores for language: {lang} =====")
        print_filtered_model_comparison_table(metrics_map, lang, print, sort_by, simplified_metrics)
    for model, data in metrics_map.items():
        # For model-specific tables, use verbose name in header if requested
        print(f"\n===== Scores for model: {model} =====")
        print_filtered_language_comparison_table(model, data, langs, print, sort_by, simplified_metrics)
    
    # By problem type
    for problem_type in ("functional", "io"):
        print(f"\n===== Scores by problem type: {problem_type} =====")
        if aggregate:
            # Recompute metrics for this problem type
            count = sum(1 for n in problem_names if get_problem_type(n) == problem_type)
            total = len(problem_names)
            percent = count / total if total else 0
            if count > 0:
                metrics_maps_category = []
                for stats_entry in model_stats_maps:
                    filtered_problems = [n for n in stats_entry["problem_names"] if get_problem_type(n) == problem_type]
                    m_stats_map = stats_entry["model_stats_map"]
                    c_rt_stats = stats_entry["canonical_runtime_stats"]
                    c_mem_stats = stats_entry["canonical_memory_stats"]
                    metrics_map_entry = {}
                    for model in m_stats_map:
                        if model.startswith("canonical_") or (models and model not in models):
                            continue
                        metrics_map_entry[model] = compute_model_metrics(filtered_problems, langs, model, m_stats_map[model], c_rt_stats, c_mem_stats)
                    metrics_maps_category.append(metrics_map_entry)
                merged_metrics = merge_metrics_maps(metrics_maps_category)
                print_filtered_model_comparison_table(merged_metrics, "avg", print, sort_by, simplified_metrics)
            else:
                print(f"No data available for problem type: {problem_type}")
        else:
            # Original behavior for single directory
            filtered = {}
            for model in metrics_map:
                # Skip models not in the specified list
                if models and model not in models:
                    continue
                filtered[model] = compute_model_metrics(
                    [n for n in problem_names if get_problem_type(n) == problem_type],
                    langs, model, model_stats_map[model],
                    canonical_runtime_stats, canonical_memory_stats
                )
            if filtered:
                count = sum(1 for n in problem_names if get_problem_type(n) == problem_type)
                total = len(problem_names)
                percent = count / total if total else 0
                print_filtered_model_comparison_table(filtered, "avg", print, sort_by, simplified_metrics)
            else:
                print(f"No data available for problem type: {problem_type}")
    
    # By problem source
    problem_sources = set(get_problem_source(n) for n in problem_names)
    for source in problem_sources:
        print(f"\n===== Scores by problem source: {source} =====")
        if aggregate:
            # Recompute metrics for this problem source
            count = sum(1 for n in problem_names if get_problem_source(n) == source)
            total = len(problem_names)
            percent = count / total if total else 0
            if count > 0:
                metrics_maps_category = []
                for stats_entry in model_stats_maps:
                    filtered_problems = [n for n in stats_entry["problem_names"] if get_problem_source(n) == source]
                    m_stats_map = stats_entry["model_stats_map"]
                    c_rt_stats = stats_entry["canonical_runtime_stats"]
                    c_mem_stats = stats_entry["canonical_memory_stats"]
                    metrics_map_entry = {}
                    for model in m_stats_map:
                        if model.startswith("canonical_") or (models and model not in models):
                            continue
                        metrics_map_entry[model] = compute_model_metrics(filtered_problems, langs, model, m_stats_map[model], c_rt_stats, c_mem_stats)
                    metrics_maps_category.append(metrics_map_entry)
                merged_metrics = merge_metrics_maps(metrics_maps_category)
                print_filtered_model_comparison_table(merged_metrics, "avg", print, sort_by, simplified_metrics)
            else:
                print(f"No data available for problem source: {source}")
        else:
            # Original behavior for single directory
            filtered = {}
            for model in metrics_map:
                # Skip models not in the specified list
                if models and model not in models:
                    continue
                filtered[model] = compute_model_metrics(
                    [n for n in problem_names if get_problem_source(n) == source],
                    langs, model, model_stats_map[model],
                    canonical_runtime_stats, canonical_memory_stats
                )
            if filtered:
                count = sum(1 for n in problem_names if get_problem_source(n) == source)
                total = len(problem_names)
                percent = count / total if total else 0
                print_filtered_model_comparison_table(filtered, "avg", print, sort_by, simplified_metrics)
            else:
                print(f"No data available for problem source: {source}")


def print_table_simple(headers: list[str], rows: list[list[str]]) -> None:
    """Prints a table with separators, header, and rows using stdout."""
    col_widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows)) + 2
        for i in range(len(headers))
    ]
    header_line = "".join(str(headers[i]).ljust(col_widths[i]) for i in range(len(headers)))
    separator = "-" * len(header_line)
    print(separator)
    print(header_line)
    print(separator)
    for row in rows:
        line = "".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
        print(line)


def print_table_latex(headers: list[str], rows: list[list[str]]) -> None:
    """Print table in LaTeX format."""
    print("\\toprule")
    header_line = " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\"
    print(header_line)
    print("\\midrule")
    for row in rows:
        escaped_cells = [str(cell).replace("%", "\\%") for cell in row]
        row_line = " & ".join(escaped_cells) + " \\\\"
        print(row_line)
    print("\\bottomrule")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Dispatch to appropriate table printing function."""
    if TABLE_FORMAT == "simple":
        print_table_simple(headers, rows)
    elif TABLE_FORMAT == "latex":
        print_table_latex(headers, rows)
    else:
        raise ValueError(f"Unknown table format: {TABLE_FORMAT}")


def print_comparison_table(
    title_col: str,
    keys: list[str],
    get_metrics: Callable[[str], dict[str, float]],
    all_metrics: Iterable[str],
    metrics_display: dict[str, str],
    sort_by: Optional[str] = None,
    row_type: str = "model",
    enrich_metrics: bool = True
) -> None:
    """Print a comparison table for the given keys and metrics."""
    metrics = [m for m in metrics_display if m in all_metrics]
    headers = [title_col] + [metrics_display[m] for m in metrics]
    table = []
    for k in keys:
        # Determine display key based on row type
        if row_type == "model":
            display_key = MODEL_CONFIG.get(k, {}).get("verbose_name", k)
        elif row_type == "language":
            display_key = EFFIBENCH_REGISTRY.get(k, {}).get("verbose_name", k)
        else:
            display_key = k
        stats = get_metrics(k) or {}
        row = [display_key]
        for m in metrics:
            val = stats.get(m, 0)
            if isinstance(val, tuple):  # Aggregated value (avg, min, max)
                avg, mn, mx = val
                if m == "pass_rate":
                    cell = f"{avg*100:.2f}%"
                else:
                    cell = f"{avg*100:.2f}% ({mn*100:.2f}%, {mx*100:.2f}%)"
            else:
                cell = f"{val*100:.2f}%"
            row.append(cell)
        table.append((k, row))
    if sort_by and sort_by in metrics:
        idx = metrics.index(sort_by) + 1
        table.sort(key=lambda kr: extract_float(kr[1][idx]), reverse=True)
    elif row_type == "model":
        # Sort models by the order in the model config
        model_order = {model: idx for idx, model in enumerate(MODEL_CONFIG.keys())}
        table.sort(key=lambda kr: model_order.get(kr[0], float('inf')))
    else:
        table.sort(key=lambda kr: kr[0])
    rows = [r for _, r in table]
    # Enrich metrics formatting for LaTeX: bold best, underline second best
    if enrich_metrics and TABLE_FORMAT == "latex":
        num_cols = len(headers)
        # gather numeric values per metric column
        numeric_cols: dict[int, list[float]] = {}
        for i in range(1, num_cols):
            vals: list[float] = []
            for row in rows:
                try:
                    vals.append(extract_float(row[i]))
                except (ValueError, IndexError):
                    vals.append(float("-inf"))
            numeric_cols[i] = vals
        best_vals: dict[int, float] = {}
        second_vals: dict[int, Optional[float]] = {}
        for i, vals in numeric_cols.items():
            uniq = sorted(set(vals), reverse=True)
            best_vals[i] = uniq[0] if uniq else float("-inf")
            second_vals[i] = uniq[1] if len(uniq) > 1 else None
        # apply formatting
        for ridx, row in enumerate(rows):
            for i in range(1, num_cols):
                val = numeric_cols[i][ridx]
                if val == best_vals[i]:
                    row[i] = f"\\textbf{{{row[i]}}}"
                elif second_vals[i] is not None and val == second_vals[i]:
                    row[i] = f"\\underline{{{row[i]}}}"
    print_table(headers, rows)


def print_filtered_model_comparison_table(
    model_data: dict,
    key: str,
    log,
    sort_by: str,
    metrics_filter: dict
) -> None:
    """Print a table comparing models for a specific language or overall with filtered metrics."""
    is_overall = key == "avg"
    if not model_data:
        print("No model data available")
        return
    first_model = next(iter(model_data.values()))
    all_metrics = first_model["avg"].keys() if is_overall else first_model["per_lang"][key].keys()
    keys = list(model_data.keys())
    def get_metrics_func(k: str) -> dict[str, float]:
        stats = model_data[k]["avg"] if is_overall else model_data[k]["per_lang"].get(key)
        return stats or {}
    print_comparison_table("Model", keys, get_metrics_func, all_metrics, metrics_filter, sort_by, row_type="model")


def print_filtered_language_comparison_table(model: str, model_data: dict, langs: list, log, sort_by: str, metrics_filter: dict):
    """Print a table comparing languages for a specific model with filtered metrics."""
    if "per_lang" not in model_data or not model_data["per_lang"]:
        print(f"No language data found for {model}")
        return
    all_metrics = next(iter(model_data["per_lang"].values())).keys()
    keys = [lang for lang in langs if lang in model_data["per_lang"]]
    def get_metrics_func(l: str) -> dict[str, float]:
        return model_data["per_lang"].get(l) or {}
    print_comparison_table("Language", keys, get_metrics_func, all_metrics, metrics_filter, sort_by, row_type="language")


def extract_float(cell: str) -> float:
    """Extract the first float from a string."""
    matches = re.findall(r"[-+]?\d*\.?\d+", cell)
    return float(matches[0]) if matches else float("-inf")


def merge_metrics_maps(maps: list) -> dict:
    """Merge multiple metrics maps into one with aggregated (avg, min, max) values."""
    if not maps:
        return {}
    
    result = deepcopy(maps[0])  # Create a structure clone
    for model in result:
        # Average section
        for k in result[model]["avg"]:
            values = [m[model]["avg"][k] for m in maps if model in m]
            result[model]["avg"][k] = (statistics.mean(values), min(values), max(values))
        
        # Per-language section
        for lang in result[model]["per_lang"]:
            for k in result[model]["per_lang"][lang]:
                values = [m[model]["per_lang"][lang][k] for m in maps if model in m and lang in m[model]["per_lang"]]
                result[model]["per_lang"][lang][k] = (statistics.mean(values), min(values), max(values))
    
    return result


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nCTRL+C detected. Exiting gracefully...", flush=True)
        os._exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    cli(standalone_mode=False)
