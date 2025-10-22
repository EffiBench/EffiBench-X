import os
import json
import click
import tempfile
from tqdm import tqdm
from pathlib import Path

from huggingface_hub import HfApi
import datasets
from datasets import load_dataset, Dataset

def _type_name(value):
    return "none" if value is None else type(value).__name__

def _collect_type_stats(records, max_records=None):
    stats = {}
    limit = len(records)
    for i in range(limit):
        obj = records[i]
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            t = _type_name(v)
            if k not in stats:
                stats[k] = {}
            stats[k][t] = stats[k].get(t, 0) + 1
    return stats


@click.group()
def cli():
    """CLI for uploading and downloading dataset to/from Hugging Face Hub."""
    pass

@cli.command()
@click.argument("dataset_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True), default="data/dataset", required=False)
@click.option("--repo-id", "-r", type=str, default="EffiBench/effibench-x", help="Hugging Face dataset repository id.")
@click.option("--token", "-t", type=str, help="Hugging Face API token. By default reads HUGGINGFACEHUB_API_TOKEN env var.")
@click.option("--inspect-only", is_flag=True, default=False, help="Inspect field types and exit without uploading.")
def upload(dataset_dir, repo_id, token, inspect_only):
    """Upload local dataset directory to Hugging Face Hub."""
    # Merge all JSON files into a single JSONL
    files = sorted(dataset_dir.glob("*.json"))
    if not files:
        click.echo(f"No JSON files found in {dataset_dir}")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        all_data = []
        for f in tqdm(files, desc="Packing dataset files"):
            obj = json.load(open(f))
            tmp.write(json.dumps(obj) + "\n")
            all_data.append(obj)
        tmp_path = tmp.name

    # Inspect schema to detect mixed-type fields
    stats = _collect_type_stats(all_data)
    mixed_keys = []
    for k, counts in stats.items():
        non_none_types = [t for t in counts.keys() if t != "none"]
        if len(non_none_types) > 1:
            mixed_keys.append(k)

    if inspect_only:
        click.echo("Field type summary (scanned all records):")
        for k in sorted(stats.keys()):
            counts = stats[k]
            parts = [f"{t}:{n}" for t, n in sorted(counts.items(), key=lambda x: x[0])]
            flag = " [MIXED]" if k in mixed_keys else ""
            click.echo(f"- {k}: {' | '.join(parts)}{flag}")
        os.remove(tmp_path)
        return

    if mixed_keys:
        click.echo("Warning: Detected mixed-type fields which may cause Arrow type errors: " + ", ".join(sorted(mixed_keys)))

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    
    # Create a dataset with a test split
    dataset = Dataset.from_list(all_data)
    
    click.echo(f"Uploading dataset to {repo_id}...")
    dataset.push_to_hub(
        repo_id=repo_id,
        split="test",
        token=token,
        max_shard_size="200MB"
    )
    
    os.remove(tmp_path)
    click.echo("Upload completed successfully.")

@cli.command()
@click.argument("dataset_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default="data/dataset", required=False)
@click.option("--repo-id", "-r", type=str, default="EffiBench/effibench-x", help="Hugging Face dataset repository id.")
@click.option("--token", "-t", type=str, help="Hugging Face API token. By default reads HUGGINGFACEHUB_API_TOKEN env var.")
def download(dataset_dir, repo_id, token):
    """Download dataset from Hugging Face Hub to local directory."""
    # Prepare local directory
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Loading dataset {repo_id} using 🤗 datasets library...")
    dataset = load_dataset(repo_id, split='test', use_auth_token=token)
    click.echo("Unpacking dataset into individual JSON files...")
    for data in tqdm(dataset, desc="Unpacking dataset"):
        source = data.get("source", "")
        problem_id = data.get("id", "")
        title_slug = data.get("title_slug", "")
        if not (source and problem_id and title_slug):
            continue
        filename = f"{source}_{problem_id}_{title_slug}.json"
        out_file = dataset_dir / filename
        with open(out_file, "w") as f:
            json.dump(data, f, indent=4)
    click.echo("Download and unpack completed successfully.")

if __name__ == "__main__":
    cli()
