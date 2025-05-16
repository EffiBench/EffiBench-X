import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from time import time

import click
from ruamel.yaml import YAML
from tqdm import tqdm

from effibench.llm import get_lm_client
from effibench.prompts import prompt_solve_problem_functional, prompt_solve_problem_io
from effibench.utils import (
    EFFIBENCH_LANGS, load_json_file, save_json_file, create_logger,
    extract_code_blocks, postprocess_solution, sort_problem_files,
)


@click.group()
def cli():
    """CLI for generating solutions using LLMs."""
    pass


def solve_problem(problem_name: str, problem_data: dict, output_file: Path, model: str, lang: str):
    """Solve a specific problem in a specific language using the specified model."""
    log = create_logger(model, problem_name, lang)
    lm_client = get_lm_client()
    
    # Load problem description
    problem_description = problem_data.get("description_md")
    if not problem_description:
        log(f"'description_md' not found in problem data", 'skip', fg="red")
        return

    # Get problem type (functional or io)
    problem_type = problem_data.get("type", "functional")

    # For functional type problems, verify starter code exists
    if problem_type == "functional":
        starter_code = problem_data.get("starter_code", {}).get(lang, None)
        if not starter_code:
            log(f"'starter_code' dictionary not found or empty in problem data", 'skip', fg="red")
            return
    # Generate solution based on problem type
    try:
        if problem_type == "functional":
            prompt = prompt_solve_problem_functional(problem=problem_description, target_lang=lang, starter_code=starter_code)
        else:  # io type
            prompt = prompt_solve_problem_io(problem=problem_description, target_lang=lang)

        llm_resp, _ = lm_client.generate(prompt=prompt, model=model)
        if not llm_resp:
            log(f"No response from {model}", "error", fg="red")
            raise Exception("No response from model")
        
        code_blocks: list[dict[str, str]] = extract_code_blocks(llm_resp)
        if len(code_blocks) == 0:
            solution_code = None
        else:
            solution_code = code_blocks[-1]["code"]
            
            if problem_type == "functional":
                solution_code = postprocess_solution(solution_code, lang)
            
            if not solution_code.strip():
                solution_code = None
            
    except Exception as e:
        log(f"Failed to generate solution: {repr(e)}", "error", fg="red")
        return

    # Save solution
    log("Solution generated successfully", fg="green")
    save_json_file(output_file, {"code": solution_code, "raw": llm_resp})


def generate_model_solution(problem_dir: Path, output_dir: Path, model: str, n_workers: int, langs: str, overwrite: bool, dry_run: bool = False) -> None:
    """Prepare all tasks for a model, log stats, and submit work to the executor."""
    log = create_logger()
    langs = [lang.strip() for lang in langs.split(",") if lang.strip() in EFFIBENCH_LANGS] if langs else EFFIBENCH_LANGS
    
    # Ensure output directory exists and is model-specific
    if output_dir.name != model:
        output_dir = output_dir / model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify which tasks need to be processed
    output_files = list(output_dir.glob("*.json"))
    output_files_names = [f.stem for f in output_files]
    
    problem_files = sort_problem_files(list(problem_dir.glob("*.json")))
    num_problems = len(problem_files)
    
    # Load all problem data at once to avoid repeated file reads
    all_problem_data = {f.stem: load_json_file(f) for f in problem_files}
    
    # Generate tasks list
    tasks = []
    for f in problem_files:
        problem_name = f.stem
        for lang in langs:
            if overwrite or f"{problem_name}_{lang}" not in output_files_names:
                output_file = output_dir / f"{problem_name}_{lang}.json"
                tasks.append((problem_name, lang, output_file))
    
    # Log statistics
    log(f"Processing {num_problems} problems, {num_problems * len(langs)} tasks in total, {len(tasks)} tasks remaining...", model, fg="blue")
    
    if dry_run:
        return
    
    if not tasks:
        return
    
    # Submit work to executor
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        submit_gap = 1.0 / n_workers
        futures = {}
        for problem_name, lang, output_file in tasks:
            future = executor.submit(solve_problem, problem_name, all_problem_data[problem_name], output_file, model, lang)
            futures[future] = (problem_name, lang)
            time.sleep(submit_gap)
        for future in as_completed(futures):
            try:
                problem_name, lang = futures[future]
                future.result()
            except Exception as e:
                log(f"Generate solution failed: {repr(e)}", model, problem_name, lang, fg="red")


@cli.command()
@click.argument("problem_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True), default=Path("data") / "dataset")
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=Path("data") / "solutions")
@click.option("--config", "-c", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), default="model_config.yaml")
@click.option("--model", "-m", help="Name of the LLM model to use.")
@click.option("--n-workers", "-w", type=int, default=1, help="Number of parallel workers for processing problems.")
@click.option("--langs", "-l", type=str, help="Comma-separated list of languages to include (e.g., 'python3,javascript'). If empty, all languages will be included.")
@click.option("--overwrite", "-f", type=bool, default=False, help="Overwrite existing solutions.")
@click.option("--dry-run", "-d", is_flag=True, help="Only show what would be done without actually generating solutions.")
def generate(problem_dir: Path, output_dir: Path, config: Path, model: str, n_workers: int, langs: str, overwrite: bool, dry_run: bool) -> None:
    """Generate solutions using models specified in config file or directly via --model."""    
    if model:
        generate_model_solution(problem_dir, output_dir, model, n_workers, langs, overwrite, dry_run)
    elif config:
        model_config = YAML().load(config.read_text())
        if not model_config:
            return
        
        with ProcessPoolExecutor(max_workers=len(model_config)) as executor:
            futures = {
                executor.submit(
                    generate_model_solution, 
                    problem_dir, 
                    output_dir, 
                    model, 
                    model_config[model].get("rate_limit", 1) or 1,
                    langs, 
                    overwrite, 
                    dry_run
                ): model
                for model in model_config.keys()
            }
            for future in as_completed(futures):
                future.result()


@cli.command()
@click.argument("problem_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True), default=Path("data") / "dataset")
@click.argument("solution_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True), default=Path("data") / "solutions")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None)
@click.option("--model", "-m", type=str, help="Name of the model to merge solutions for.")
@click.option("--langs", "-l", type=str, help="Comma-separated list of languages to include (e.g., 'python3,javascript'). If empty, all languages will be included.")
def merge(problem_dir: Path, solution_dir: Path, output_dir: Path, model: str, langs: str) -> None:
    """Merge solutions from all models into single files by model name."""
    log = create_logger()
    langs = [lang.strip() for lang in langs.split(",") if lang.strip() in EFFIBENCH_LANGS] if langs else EFFIBENCH_LANGS
    
    if output_dir is None:
        output_dir = solution_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_files = sort_problem_files(list(problem_dir.glob("*.json")))
    
    model_dirs = [d for d in solution_dir.iterdir() if d.is_dir()]
    log(f"Merging {len(model_dirs)} models...", fg="blue")
    
    for model_dir in model_dirs:
        if model is not None and model_dir.name != model:
            continue
        
        current_model_name = model_dir.name
        
        merged_data = defaultdict(dict)
        for problem_file in tqdm(problem_files, desc=f"Processing {current_model_name}", leave=False):
            for lang in langs:
                solution_file = model_dir / f"{problem_file.stem}_{lang}.json"
                merged_data[problem_file.stem][lang] = load_json_file(solution_file)["code"] if solution_file.exists() else None
        
        output_file = output_dir / f"{current_model_name}.json"
        save_json_file(output_file, dict(merged_data))
        log(f"Merged {len(merged_data)} problems for {current_model_name}", fg="green")


@cli.command()
@click.argument("problem_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True), default=Path("data") / "dataset")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=Path("data") / "solutions")
@click.option("--langs", "-l", type=str, help="Comma-separated list of languages to include (e.g., 'python3,javascript'). If empty, all languages will be included.")
def merge_canonical_solutions(problem_dir: Path, output_dir: Path, langs: str) -> None:
    """Merge canonical solutions from all models into single files by model name."""
    log = create_logger()
    langs = [lang.strip() for lang in langs.split(",") if lang.strip() in EFFIBENCH_LANGS] if langs else EFFIBENCH_LANGS

    output_dir.mkdir(parents=True, exist_ok=True)
    problem_files = sort_problem_files(list(problem_dir.glob("*.json")))

    runtime_solutions = defaultdict(dict)
    memory_solutions = defaultdict(dict)
    normal_solution = defaultdict(dict)

    for problem_file in tqdm(problem_files, desc="Processing problems", leave=False):
        problem_data = load_json_file(problem_file)
        for lang in langs:
            # Initialize all solutions to None for this problem-language pair
            runtime_solutions[problem_file.stem][lang] = None
            memory_solutions[problem_file.stem][lang] = None
            normal_solution[problem_file.stem][lang] = None
            
            lang_solution_dict = problem_data["solutions"].get(lang)
            if not lang_solution_dict:
                continue
            
            # Parse runtime distribution
            runtime_dist = json.loads(lang_solution_dict.get("runtimeDistribution", "[]"))
            runtime_solution = next((item[2] for item in runtime_dist if item[2] is not None), None)
            if runtime_solution:
                runtime_solutions[problem_file.stem][lang] = runtime_solution
            
            # Parse memory distribution
            memory_dist = json.loads(lang_solution_dict.get("memoryDistribution", "[]"))
            memory_solution = next((item[2] for item in memory_dist if item[2] is not None), None)
            if memory_solution:
                memory_solutions[problem_file.stem][lang] = memory_solution
            
            # Parse direct code solution
            code = lang_solution_dict.get("code")
            if code:
                normal_solution[problem_file.stem][lang] = code

    # Save canonical solutions
    runtime_output_file = output_dir / "canonical_runtime.json"
    memory_output_file = output_dir / "canonical_memory.json"
    normal_output_file = output_dir / "canonical_normal.json"

    save_json_file(runtime_output_file, dict(runtime_solutions))
    save_json_file(memory_output_file, dict(memory_solutions))
    save_json_file(normal_output_file, dict(normal_solution))

    log(f"Saved {len(runtime_solutions)} problems with runtime solutions", fg="green")
    log(f"Saved {len(memory_solutions)} problems with memory solutions", fg="green")
    log(f"Saved {len(normal_solution)} problems with direct code solutions", fg="green")


if __name__ == "__main__":
    cli()