import os
import random
import sys
import json
import signal
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
import dotenv
from tqdm import tqdm

from effibench.llm import get_lm_client
from effibench.prompts import (
    prompt_test_suite_functional, prompt_fix_test_runner, prompt_generate_new_test_runner,
    prompt_test_suite_io, prompt_translate_solution_io
)
from effibench.utils import (
    EFFIBENCH_LANGS,
    load_json_file, materialize_function_from_code,
    parse_distribution, retry, save_json_file, extract_code_blocks,
    get_lang_by_md_lang, execute_with_timeout, create_logger,
    postprocess_test_runner, postprocess_solution, get_full_code, sort_problem_files, parse_range, try_int
)
from effibench.run_tests import run_tests

PWD = Path(__file__).parent


def is_functional(file_stem: str) -> bool:
    return file_stem.startswith("leetcode_")


def _save_llm_messages(output_dir: Path, file_stem: str, error: str, messages: list) -> None:
    out_file = output_dir / f"{file_stem}.error"
    content = f"{error}\n\n"
    for msg in messages:
        content += f"Role: {msg['role']}\n{msg['content']}\n\n"
    out_file.write_text(content)


def extract_task_code_blocks_functional(llm_output: str, expected_langs: list[str]):
    """Extracts generator, evaluator, and test runner code blocks for functional problems."""
    code_blocks: list[dict[str, str]] = extract_code_blocks(llm_output)

    if len(code_blocks) < 3:
        raise ValueError(f"Expected at least 3 code blocks (generator, evaluator, test runner), got {len(code_blocks)} code blocks.")

    # Check generator block
    if "generate_test_cases" in code_blocks[0]["code"]:
        test_case_generator = code_blocks[0]["code"].strip()
    else:
        raise ValueError("Missing function 'generate_test_cases' in the first code block (expected generator).")

    # Check evaluator block
    if "evaluate" in code_blocks[1]["code"]:
        evaluator = code_blocks[1]["code"].strip()
    else:
        raise ValueError("Missing function 'evaluate' in the second code block (expected evaluator).")

    # Process test runner blocks
    test_runners: dict[str, str | None] = {lang: None for lang in expected_langs}
    for code_block in code_blocks[2:]:
        lang_tag = code_block["lang"].strip() if code_block["lang"] else ""
        block_content = code_block["code"].strip()
        
        if not lang_tag or get_lang_by_md_lang(lang_tag) not in expected_langs:
            raise ValueError(f"Unrecognized language tag '{lang_tag}' found.")
        lang = get_lang_by_md_lang(lang_tag)
        test_runners[lang] = postprocess_test_runner(block_content, lang)
    
    return test_case_generator, evaluator, test_runners


def extract_task_code_blocks_stdio(llm_output: str):
    """Extracts generator and evaluator code blocks for stdio problems."""
    code_blocks: list[dict[str, str]] = extract_code_blocks(llm_output)

    if len(code_blocks) < 2:
        raise ValueError(f"Expected at least 2 code blocks (generator, evaluator), got {len(code_blocks)} code blocks.")

    # Check generator block
    if "generate_test_cases" in code_blocks[0]["code"]:
        test_case_generator = code_blocks[0]["code"].strip()
    else:
        raise ValueError("Missing function 'generate_test_cases' in the first code block (expected generator).")

    # Check evaluator block
    if "evaluate" in code_blocks[1]["code"]:
        evaluator = code_blocks[1]["code"].strip()
    else:
        raise ValueError("Missing function 'evaluate' in the second code block (expected evaluator).")
    
    return test_case_generator, evaluator


def extract_test_runner(response: str, lang: str) -> str:
    code_blocks = extract_code_blocks(response)
    
    if not code_blocks or len(code_blocks) != 1:
        raise ValueError(f"Expected exactly 1 code block, got {len(code_blocks)}")
    # assert code_blocks[0]["lang"] in EFFIBENCH_REGISTRY[lang]["md_langs"], f"Expected language {lang}, got {code_blocks[0]['lang']}"
    
    test_runner = code_blocks[0]["code"]
    test_runner = postprocess_test_runner(test_runner, lang)
    
    return test_runner


def run_and_eval_test_case_generator(test_case_generator: str, num_cases: int = 100, evaluator: str | None = None) -> list[dict[str, str]]:

    generate_test_cases = materialize_function_from_code(test_case_generator, "generate_test_cases")

    timeout = 1 * num_cases
    try:
        generate_test_cases = execute_with_timeout(generate_test_cases, timeout, num_cases)

        if not isinstance(generate_test_cases, list):
            raise TypeError(f"Expected list, got {type(generate_test_cases).__name__}")
        if len(generate_test_cases) != num_cases:
            raise ValueError(f"Expected {num_cases} test cases, got {len(generate_test_cases)}")
        # Check for duplicate inputs in generated test cases
        # inputs = [tc["input"] for tc in generate_test_cases]
        # if len(set(inputs)) != len(inputs):
            # raise ValueError("Duplicate inputs found in generated test cases")
        # Evaluator self-check if evaluator provided
        if evaluator:
            eval_func = materialize_function_from_code(evaluator, "evaluate")

            for tc in generate_test_cases:
                if not eval_func(tc["output"], tc["output"]):
                    raise ValueError("Evaluator failed self-check on identical outputs")

        return generate_test_cases
    except TimeoutError as e:
        raise TimeoutError(f"Timeout ({timeout}s) generating {num_cases} test cases") from e


def process_problem(problem_file: Path, output_dir: Path, prototype_model: str, fix_model: str, target_languages: list[str]) -> None:
    problem_name = problem_file.stem
    output_file = output_dir / f"{problem_name}.json"
    progress_file = output_file.with_suffix(".progress")

    if output_file.exists():
        return
    
    lm_client = get_lm_client()
    log = create_logger(problem_name)
    pid = os.getpid()
    log(f"[{pid}] Worker started processing...", fg="blue")
    
    # Load data
    problem_data = load_json_file(problem_file)
    problem_description = problem_data.get("description_md")
    solutions = problem_data["solutions"]
    problem_type = problem_data.get("type", "functional")
    
    if problem_type == "functional":
        # Should have a Python 3 solution and solutions for all target languages
        if not (solutions.get("python3") or {}).get("code"):
            raise ValueError(f"{problem_name} does not have a Python 3 solution, skipped")
        if not all((solutions.get(lang) or {}).get("code") for lang in target_languages):
            raise ValueError(f"{problem_name} does not have solutions for {', '.join(target_languages)}, skipped")
    elif problem_type == "io":
        # Should have at least one solution
        if not any((solutions.get(lang) or {}).get("code") for lang in target_languages):
            raise ValueError(f"{problem_name} does not have any solution for {', '.join(target_languages)}, skipped")
    
    # Initialize output data
    output_data = load_json_file(progress_file) if progress_file.exists() else {
        "test_case_generator": None,
        "evaluator": None,
        "test_runners": {lang: None for lang in EFFIBENCH_LANGS},
        "generated_tests": None,
    }
    if isinstance(output_data.get("generated_tests", None), str):
        output_data["generated_tests"] = json.loads(output_data["generated_tests"])
    
    # For IO problems, keep the key but set it to None (dataset contract)
    if problem_type == "io":
        output_data["test_runners"] = None
    
    # Generate test template based on problem type
    messages = []
    last_error = None
    
    if output_data["test_case_generator"] is None:
        log(f"Generating test case generator for {problem_type} problem...", fg="blue")
        
        if problem_type == "functional":
            prompt_input = prompt_test_suite_functional(
                problem=problem_description,
                starter_code_dict={l: s for l, s in problem_data["starter_code"].items() if l in target_languages},
                solution=solutions["python3"]["code"],
                solution_lang="python3"
            )
            
            @retry(max_retries=3, backoff_factor=1, error_types=(Exception,))
            def call_and_extract_and_generate_functional():
                llm_resp, messages = lm_client.generate(prompt=prompt_input, model=prototype_model)
                test_case_generator, evaluator, test_runners = extract_task_code_blocks_functional(llm_resp, target_languages)
                generated_cases = run_and_eval_test_case_generator(
                    test_case_generator,
                    num_cases=100,
                    evaluator=evaluator,
                )
                return test_case_generator, evaluator, test_runners, generated_cases, messages

            try:
                test_case_generator, evaluator, test_runners, generated_cases, messages = call_and_extract_and_generate_functional()
                output_data["test_case_generator"] = test_case_generator
                output_data["evaluator"] = evaluator
                unvalidated_test_runners = {lang: test_runners.get(lang) for lang in target_languages if test_runners.get(lang) is not None}
                output_data["generated_tests"] = generated_cases
            except Exception as e:
                log(repr(e), fg="red")
                _save_llm_messages(output_dir, problem_file.stem, repr(e), messages)
                raise
            
        else:
            if solutions.get("python3") is not None:
                solution_code = solutions["python3"]["code"]
                solution_lang = "python3"
            else:
                for lang in target_languages:
                    if (solutions.get(lang) or {}).get("code"):
                        solution_code = solutions[lang]["code"]
                        solution_lang = lang
                        break
            
            log(f"Using {solution_lang} solution for generating test suite", fg="blue")
            prompt_input = prompt_test_suite_io(
                problem=problem_description,
                solution=solution_code,
                solution_lang=solution_lang
            )
            
            # Replace @retry decorator with a simple for loop
            max_retries = 3
            last_error = None  # Initialize last_error before the loop
            for retry_count in range(max_retries):
                try:
                    if retry_count == 0:
                        # First attempt - generate from scratch
                        llm_resp, messages = lm_client.generate(prompt=prompt_input, model=prototype_model)
                    else:
                        error_traceback = traceback.format_exc()
                        error_message = f"{repr(last_error)}\n\nTraceback:\n{error_traceback}"
                        fix_prompt = f"""
The code you provided for the test case generator and evaluator has an error:

{error_message}

Please fix the code for both the test case generator and evaluator to resolve this issue.
Make sure the generator creates valid test cases and the evaluator correctly assesses solutions.
Respond two code blocks, one for the test case generator and one for the evaluator.
"""
                        messages.append({"role": "user", "content": fix_prompt})
                        llm_resp, messages = lm_client.generate(messages=messages, model=prototype_model)
                    
                    log(f"Generated test case generator and evaluator", fg="blue")
                    test_case_generator, evaluator = extract_task_code_blocks_stdio(llm_resp)
                    log(f"Extracted test case generator and evaluator", fg="blue")
                    generated_cases = run_and_eval_test_case_generator(
                        test_case_generator,
                        num_cases=100,
                        evaluator=evaluator,
                    )
                    log(f"Generated {len(generated_cases)} test cases", fg="blue")
                    
                    output_data["test_case_generator"] = test_case_generator
                    output_data["evaluator"] = evaluator
                    output_data["generated_tests"] = generated_cases
                    
                    # End-to-end validation for stdio solutions
                    log(f"Validating end-to-end solutions for {', '.join(target_languages)}", fg="blue")
                    for lang in target_languages:
                        log("Validating solution...", lang, fg="blue")
                        if (solutions.get(lang) or {}).get("code") is None:
                            log("Skipping because it has no solution", lang, fg="yellow")
                            continue
                        log("Running tests...", lang, fg="blue")
                        run_tests(
                            lang=lang,
                            solution=solutions[lang]["code"],
                            test_cases=generated_cases,
                            evaluator=evaluator,
                            test_runner=None,
                            as_batch=False,
                        )
                        log("Solution end-to-end verified", lang, fg="green")
                    
                    # No test runners for stdio problems
                    unvalidated_test_runners = {}
                    
                    # Success - break out of retry loop
                    break
                    
                except Exception as e:
                    last_error = e
                    log(f"Attempt {retry_count + 1}/{max_retries} failed: {repr(e)}", fg="red")
                    _save_llm_messages(output_dir, f"{problem_file.stem}_attempt_{retry_count}", repr(e), messages)
                    
                    # If this is the last attempt, re-raise the exception
                    if retry_count == max_retries - 1:
                        raise
    else:
        # If we already have a test case generator, we might need to validate test runners for functional problems
        unvalidated_test_runners = {
            lang: None for lang in target_languages 
            if problem_type == "functional" and output_data["test_runners"][lang] is None
        }
    
    # Validate test runners for functional problems only
    if problem_type == "functional":
        for lang in unvalidated_test_runners.keys():
            forked_messages = [m for m in messages] # Fork the messages to avoid conflicts between languages
            
            if unvalidated_test_runners[lang] is None:
                # Generate a new test runner based on the context.
                log("No test runner found. Generating a new one.", lang, fg="blue")
                assert any(output_data["test_runners"][l] for l in target_languages), "No test runners found for any language"
                prompt = prompt_generate_new_test_runner(
                    lang,
                    problem=problem_description,
                    starter_code=problem_data["starter_code"][lang],
                    test_case_generator=output_data["test_case_generator"],
                    evaluator=output_data["evaluator"], 
                    test_runners=output_data["test_runners"]
                )
                response, forked_messages = lm_client.generate(messages=forked_messages + [{"role": "user", "content": prompt}], model=fix_model)
                unvalidated_test_runners[lang] = extract_test_runner(response, lang)

            test_runner = unvalidated_test_runners[lang] # Get the test runner

            try:
                run_tests(
                    lang=lang,
                    solution=solutions[lang]["code"],
                    test_cases=output_data["generated_tests"],
                    evaluator=output_data["evaluator"],
                    test_runner=test_runner,
                )
                log("Test runner verified", lang, fg="green")
                output_data["test_runners"][lang] = test_runner
                save_json_file(progress_file, output_data)
                continue
            except Exception as e:
                last_error = repr(e)
                log(f"Test runner failed: {e}", lang, fg="red")
                _save_llm_messages(output_dir, f"{problem_file.stem}_{lang}_attempt_0", repr(e), forked_messages)
            
            for i in range(3):
                try:
                    fix_prompt = prompt_fix_test_runner(lang=lang, err_msg=last_error, test_runner=test_runner, full_code=get_full_code(lang, solutions[lang]["code"], test_runner))
                    forked_messages.append({"role": "user", "content": fix_prompt})
                    fixed_resp, forked_messages = lm_client.generate(messages=forked_messages, model=fix_model)

                    test_runner = extract_test_runner(fixed_resp, lang)
                    run_tests(
                        lang=lang,
                        solution=solutions[lang]["code"],
                        test_cases=output_data["generated_tests"],
                        evaluator=output_data["evaluator"],
                        test_runner=test_runner,
                    )
                    output_data["test_runners"][lang] = test_runner
                except Exception as e:
                    log(f"Failed to fix test runner: {e}", lang, fg="red")
                    _save_llm_messages(output_dir, f"{problem_file.stem}_{lang}_attempt_{i+1}", repr(e), forked_messages)
    
    # Check if need to save the output
    if problem_type == "functional":
        all_verified = all(output_data["test_runners"].get(lang) for lang in target_languages)
        some_verified = any(output_data["test_runners"].get(lang) for lang in target_languages)
    else:
        all_verified = output_data["test_case_generator"] is not None and output_data["evaluator"] is not None
        some_verified = all_verified
    
    output_data["generated_tests"] = json.dumps(output_data["generated_tests"])
    
    if all_verified:
        log(f"All {problem_type} requirements verified successfully", fg="green")
        save_json_file(output_file, output_data)
        progress_file.unlink(missing_ok=True)
    elif some_verified:
        if problem_type == "functional":
            log(f"Some languages verified: {', '.join(lang for lang in target_languages if output_data['test_runners'][lang])}", fg="yellow")
        else:
            log("Partial verification completed", fg="yellow")
        save_json_file(progress_file, output_data)
    else:
        log("No verification completed", fg="red")


@click.group()
def cli():
    pass

@cli.command()
@click.argument("problem-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True), default=PWD / "data" / "problems")
@click.argument("output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=PWD / "data" / "templates")
@click.option("--prototype-model", "-pm", type=str, default="gemini-2.5-pro-exp-03-25", help="Model used for initial test case generation.")
@click.option("--fix-model", "-fm", type=str, default="gemini-2.5-pro-exp-03-25", help="Model used for fixing test runners.")
@click.option("--langs", type=str, help="Comma-separated list of languages to focus on (e.g., 'python3,javascript'). If empty, all languages will be processed.")
@click.option("--range", type=str, help="Range of problem IDs to process in format 'a:b'. If 'a' is omitted, start from 0. If 'b' is omitted, process until the end.")
@click.option("--reverse", is_flag=True, help="Process problems in reverse order.")
@click.option("--shuffle", is_flag=True, help="Shuffle the problem files.")
@click.option("--include", type=str, default="*.json", help="File pattern to include.")
@click.option("--exclude", type=str, help="File pattern to exclude.")
@click.option("--skip-initialization", is_flag=True, help="Skip the initialization step.")
@click.option("--n-workers", default=1, help="Number of parallel worker processes.")
def generate(problem_dir: Path, output_dir: Path, prototype_model: str, fix_model: str, n_workers: int, langs: str = None, range: str = None, reverse: bool = False, shuffle: bool = False, include: str = "*.json", exclude: str = None, skip_initialization: bool = False) -> None:
    log = create_logger()
    
    # Log main process ID
    main_pid = os.getpid()
    log(f"Main process PID: {main_pid}", fg="blue")
    
    # Parse the languages option
    target_langs = [lang.strip() for lang in langs.split(",") if lang.strip() in EFFIBENCH_LANGS] if langs else EFFIBENCH_LANGS
    
    # Parse range if provided
    start, end = parse_range(range)
    
    # Scan for files to process
    problem_files = list(problem_dir.glob(include))
    if exclude:
        exclude_files = list(problem_dir.glob(exclude))
        problem_files = [f for f in problem_files if f not in exclude_files]
    
    # Filter by problem ID range
    if start > 0 or end < int(1e9):
        problem_files = [f for f in problem_files if start <= try_int(f.stem.split("_")[1]) <= end]
        log(f"Filtered to {len(problem_files)} input files within ID range {start}-{end}", fg="blue")
    
    # Create the output directory and scan for previously generated output files to preprocess
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = [f for f in output_dir.glob("*.*") if f.suffix in [".json", ".progress"]]
    
    if not skip_initialization:
        # [Functional Problems] Rename files to .json if all languages have test runners, .progress otherwise
        for output_file in tqdm(sort_problem_files(output_files), desc="Initializing functional problems"):
            if not is_functional(output_file.stem):
                continue
            
            output_data = load_json_file(output_file)
            # Functional problems: `test_runners` exists and is a dict (per dataset contract)
            if "test_runners" in output_data: # functional problems
                suffix = ".json" if all(output_data["test_runners"].get(lang) for lang in target_langs) else ".progress"
                if output_file.suffix != suffix:
                    output_file.rename(output_file.with_suffix(suffix))
    
    # Filter out finished problems
    output_file_stem_set = set(f.stem for f in output_dir.glob("*.json"))
    problem_files = [f for f in problem_files if f.stem not in output_file_stem_set]
    
    # Sort the problem files
    problem_files = sort_problem_files(problem_files)
    if reverse:
        problem_files.reverse()
    if shuffle:
        random.shuffle(problem_files)
    
    log(f"Starting processing {len(problem_files)} problems with {n_workers} workers", fg="blue")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_problem, f, output_dir, prototype_model, fix_model, target_langs): f
            for f in problem_files
        }
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log(f"{futures[future].stem} failed: {e}", fg="red")
        except KeyboardInterrupt:
            log("Keyboard interrupt received. Cancelling remaining tasks...", fg="yellow")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False)
            log("Executor shutdown complete", fg="blue")
            sys.exit(1)


def translate_io_problem(problem_file: Path, template_dir: Path, output_dir: Path, prototype_model: str, fix_model: str, target_langs: list[str]) -> None:
    problem_name = problem_file.stem
    template_file = template_dir / f"{problem_name}.json"
    output_file = output_dir / f"{problem_name}.json"
    progress_file = output_file.with_suffix(".progress")
    
    if output_file.exists():
        return
    
    lm_client = get_lm_client()
    log = create_logger(problem_name)
    pid = os.getpid()
    log(f"Worker process PID: {pid}", fg="blue")
    
    problem_data = load_json_file(problem_file)
    template_data = load_json_file(template_file)
    output_data = load_json_file(progress_file) if progress_file.exists() else problem_data | template_data
    
    generated_tests = template_data["generated_tests"]
    if isinstance(generated_tests, str):
        generated_tests = json.loads(generated_tests)
    
    def find_best(dist):
        return next((d for d in dist if d[2] is not None), None)

    def find_matching_code(dist, code):
        return next((d for d in dist if d[2] == code), None)
    
    best = {"runtime": None, "memory": None}
    
    for lang in ["python3"] + target_langs:
        if not (problem_data["solutions"].get(lang) or {}).get("code"):
            continue
            
        runtime_dist = parse_distribution(problem_data["solutions"][lang]["runtimeDistribution"], "runtime", lang, log)
        memory_dist = parse_distribution(problem_data["solutions"][lang]["memoryDistribution"], "memory", lang, log)
        
        best_runtime = find_best(runtime_dist)
        best_memory = find_best(memory_dist)
        
        if best_runtime and best_memory:
            best["runtime"], best["memory"] = best_runtime, best_memory
            refer_lang = lang
            break
    
    if best["runtime"] is None or best["memory"] is None:
        log(f"No best runtime or memory found for any language", fg="yellow")
        return
    
    is_same_code = best["runtime"][2] == best["memory"][2]
    matching_opposite = {
        "runtime": find_matching_code(memory_dist, best["runtime"][2]),
        "memory": find_matching_code(runtime_dist, best["memory"][2])
    }
    
    for lang in target_langs:
        if (output_data["solutions"].get(lang) or {}).get("code"):
            continue
        
        solution_data = {}
        runtime_dist = []
        memory_dist = []
        
        # Retry to generate a working solution
        max_retries = 5
        for attempt in range(max_retries):
            try:
                for dist_type in ["runtime", "memory"]:
                    if dist_type == "memory" and is_same_code:
                        continue
                    
                    reference_solutions = {refer_lang: best[dist_type][2]}
            
                    # First attempt or generate a new solution based on feedback
                    if attempt == 0:
                        prompt = prompt_translate_solution_io(
                            problem=problem_data["description_md"],
                            target_lang=lang,
                            reference_solutions=reference_solutions
                        )
                        response, messages = lm_client.generate(prompt=prompt, model=prototype_model)
                    else:
                        # Fix attempt with error message
                        error_traceback = traceback.format_exc()
                        error_message = f"{repr(last_error)}\n\nTraceback:\n{error_traceback}"
                        
                        fix_prompt = f"""
The solution you provided for {lang} failed the test cases with the following error:

{error_message}

Please fix the solution for the problem. Make sure the solution correctly handles all test cases.
Respond with the fixed solution code block in {lang}.
"""
                        messages.append({"role": "user", "content": fix_prompt})
                        response, messages = lm_client.generate(messages=messages, model=fix_model)
                        log(f"Attempt {attempt+1}/{max_retries} to fix {lang} solution for {dist_type}", fg="blue")
                    
                    code_blocks = extract_code_blocks(response)
                    if not code_blocks or not code_blocks[-1]["code"]:
                        log(f"No code blocks found for {lang}", fg="yellow")
                        raise ValueError(f"No code blocks found in the response for {lang}")
                    
                    code = code_blocks[-1]["code"]
                    
                    # Run tests with the generated code
                    log(f"Testing {lang} solution (attempt {attempt+1}/{max_retries})", fg="blue")
                    run_tests(
                        lang=lang,
                        solution=code,
                        test_cases=generated_tests,
                        evaluator=output_data["evaluator"],
                        as_batch=False,
                    )
                    log(f"Solution for {lang} ({dist_type}) passed tests on attempt {attempt+1}", fg="green")
                    
                    # If we got here, we have a working solution
                    if dist_type == "runtime":
                        runtime, memory = best["runtime"][0], matching_opposite["runtime"][0]
                        solution_data["code"] = code
                        solution_data["runtime"] = runtime
                        solution_data["memory"] = memory
                    else: # dist_type == "memory"
                        runtime, memory = matching_opposite["memory"][0], best["memory"][0]

                    runtime_dist.append([runtime, 0, code])
                    memory_dist.append([memory, 0, code])
            
                runtime_dist.sort(key=lambda x: x[0])
                memory_dist.sort(key=lambda x: x[0])
                
                solution_data["runtimeDistribution"] = json.dumps(runtime_dist)
                solution_data["memoryDistribution"] = json.dumps(memory_dist)
            
            except Exception as e:
                last_error = e
                log(f"Attempt {attempt+1}/{max_retries} failed: {repr(e)}", fg="red")
                
                # If this is the last attempt and still failing, skip this language
                if attempt == max_retries - 1:
                    log(f"Failed to generate working solution for {lang} after {max_retries} attempts", fg="red")
                    _save_llm_messages(output_dir, f"{problem_name}_{lang}_{dist_type}", repr(e), messages)
                    raise
        
        output_data["solutions"][lang] = solution_data
    
    all_verified = all((output_data["solutions"].get(lang) or {}).get("code") for lang in target_langs)
    if all_verified:
        log(f"All solutions verified successfully", fg="green")
        save_json_file(output_file, output_data)
    else:
        verified_langs = [lang for lang in target_langs if (output_data["solutions"].get(lang) or {}).get("code")]
        log(f"Partial verification completed: {len(verified_langs)}/{len(target_langs)} languages", fg="yellow")
        save_json_file(progress_file, output_data)
        

@cli.command()
@click.argument("problem-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), default=PWD / "data" / "problems")
@click.argument("template-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), default=PWD / "data" / "templates")
@click.argument("output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=PWD / "data" / "dataset")
@click.option("--prototype-model", "-pm", type=str, default="gemini-2.5-pro-exp-03-25", help="Model used for initial test case generation.")
@click.option("--fix-model", "-fm", type=str, default="gemini-2.5-pro-exp-03-25", help="Model used for fixing test runners.")
@click.option("--langs", type=str, help="Comma-separated list of languages to include (e.g., 'python3,javascript'). If empty, all languages will be included.")
@click.option("--range", type=str, help="Range of problem IDs to process in format 'a:b'. If 'a' is omitted, start from 0. If 'b' is omitted, process until the end.")
@click.option("--reverse", is_flag=True, help="Process problems in reverse order.")
@click.option("--skip-initialization", is_flag=True, help="Skip the initialization step.")
@click.option("--n-workers", default=1, help="Number of parallel worker processes.")
def finalize(problem_dir: Path, template_dir: Path, output_dir: Path, prototype_model: str, fix_model: str, langs: str, range: str, reverse: bool, skip_initialization: bool, n_workers: int) -> None:
    """Merge completed template files with problem files and save to the output directory."""
    log = create_logger()
    
    # Parse the languages option
    target_langs = [lang.strip() for lang in langs.split(",") if lang.strip() in EFFIBENCH_LANGS] if langs else EFFIBENCH_LANGS
    
    # Parse range if provided
    start, end = parse_range(range)
    
    # Scan for files to process
    problem_files = list(problem_dir.glob("*.json"))
    template_files = list(template_dir.glob("*.json"))
    template_file_stem_set = set(f.stem for f in template_files)
    problem_files = [f for f in problem_files if f.stem in template_file_stem_set]
    
    # Filter by problem ID range
    if start > 0 or end < int(1e9):
        problem_files = [f for f in problem_files if start <= try_int(f.stem.split("_")[1]) <= end]
        log(f"Filtered to {len(problem_files)} input files within ID range {start}-{end}", fg="blue")

    # Create the output directory and scan for previously generated output files to preprocess
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = [f for f in output_dir.glob("*.*") if f.suffix in [".json", ".progress"]]
    
    if not skip_initialization:
        # [IO Problems] Rename output files to .json if fully processed for the target languages, .progress otherwise
        for output_file in tqdm(sort_problem_files(output_files), desc="Initializing IO problems"):
            if is_functional(output_file.stem):
                continue
            
            output_data = load_json_file(output_file)
            # IO problems: `test_runners` must exist and be null (per dataset contract)
            if output_data.get("test_runners") is None: # io problems
                suffix = ".json" if all((output_data["solutions"].get(lang) or {}).get("code") for lang in target_langs) else ".progress"
                if output_file.suffix != suffix:
                    output_file.rename(output_file.with_suffix(suffix))

    # Filter out finished problems
    output_file_stem_set = set(f.stem for f in output_dir.glob("*.json"))
    problem_files = [f for f in problem_files if f.stem not in output_file_stem_set]

    # Sort the problem files
    problem_files = sort_problem_files(problem_files)
    if reverse:
        problem_files.reverse()
    
    log(f"Starting processing {len(problem_files)} problems with {n_workers} workers", fg="blue")
    
    functional_problem_files = [f for f in problem_files if is_functional(f.stem)]
    io_problem_files = [f for f in problem_files if not is_functional(f.stem)]
    
    for problem_file in tqdm(functional_problem_files, desc="Finalizing functional problems"):
        template_file = template_dir / f"{problem_file.stem}.json"
        
        problem_data = load_json_file(problem_file)
        template_data = load_json_file(template_file)
        
        problem_data["solutions"] = {lang: problem_data["solutions"].get(lang) for lang in target_langs}
        for lang, sol in problem_data["solutions"].items():
            if sol is not None:
                sol["code"] = postprocess_solution(sol["code"], lang)
        
        template_data["test_runners"] = {lang: postprocess_test_runner(template_data["test_runners"].get(lang), lang) for lang in target_langs}
        
        problem_data.update(template_data)
        output_file = output_dir / f"{problem_file.stem}.json"
        save_json_file(output_file, problem_data)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(translate_io_problem, f, template_dir, output_dir, prototype_model, fix_model, target_langs): f
            for f in io_problem_files
        }
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log(f"{futures[future].stem} failed: {e}", fg="red")
        except KeyboardInterrupt:
            log("Keyboard interrupt received. Cancelling remaining tasks...", fg="yellow")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False)
            log("Executor shutdown complete", fg="blue")
            sys.exit(1)

if __name__ == "__main__":    
    # Set up signal handler for cleaner exit
    def signal_handler(sig, frame):
        print("\nCTRL+C detected. Exiting gracefully...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    cli()
