import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
import typer
from typing_extensions import Annotated

load_dotenv()

def download_from_huggingface(
    repo_id: Annotated[str, typer.Option()] = "", 
    remote_dir: Annotated[str, typer.Option()] = "/",
    repo_type: Annotated[str, typer.Option()]= "model", 
    local_dir: Annotated[str, typer.Option()]=".data"
):
    """Download files or folders from Hugging Face repository.
    
    Args:
        repo_id: Repository ID on Hugging Face
        repo_dir: File or folder path within the repository to download
        repo_type: Type of the repository (e.g., "model", "dataset")
        local_dir: Local directory to download files to
    """
    
    try:
        # List all files in the repository to check if repo_dir is a file or folder
        repo_files = list_repo_files(repo_id=repo_id, repo_type=repo_type, token=os.getenv("HF_TOKEN"))
        
        # Check if repo_dir is an exact file match
        is_file = remote_dir in repo_files
        
        if is_file:
            # Download single file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_dir,
                local_dir=local_dir,
                repo_type=repo_type,
                token=os.getenv("HF_TOKEN")
            )
            print("Downloaded file to:", local_path)
        else:
            # Assume it's a folder and download all matching files
            local_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{remote_dir}/**",
                local_dir=local_dir,
                repo_type=repo_type,
                token=os.getenv("HF_TOKEN")
            )
            print("Downloaded folder to:", local_path)
            
    except Exception as e:
        print(f"Error downloading {remote_dir}: {e}")
        raise
    
    return local_path

if __name__ == "__main__":
    typer.run(download_from_huggingface)
