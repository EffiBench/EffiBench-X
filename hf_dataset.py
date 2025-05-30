import os
import json
import click
import tempfile
from tqdm import tqdm
from pathlib import Path

from huggingface_hub import HfApi
from datasets import load_dataset, Dataset

@click.group()
def cli():
    """CLI for uploading and downloading dataset to/from Hugging Face Hub."""
    pass

@cli.command()
@click.argument("dataset_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True), default="data/dataset", required=False)
@click.option("--repo-id", "-r", type=str, default="EffiBench/effibench-x", help="Hugging Face dataset repository id.")
@click.option("--token", "-t", type=str, help="Hugging Face API token. By default reads HUGGINGFACEHUB_API_TOKEN env var.")
def upload(dataset_dir, repo_id, token):
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

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    
    # Create a dataset with a test split
    dataset = Dataset.from_list(all_data)
    
    click.echo(f"Uploading dataset to {repo_id}...")
    dataset.push_to_hub(
        repo_id=repo_id,
        split="test",
        token=token,
        max_shard_size="500MB",
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
            json.dump(data, f)
    click.echo("Download and unpack completed successfully.")

if __name__ == "__main__":
    cli()
