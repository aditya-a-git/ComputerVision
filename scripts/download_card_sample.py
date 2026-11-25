import argparse
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from core.data.card import REPO, groups


def main(list_only=False):
    p=argparse.ArgumentParser()
    p.add_argument("--list-only",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--revision",default="main")
    p.add_argument("--max-files",type=int,default=5)
    p.add_argument("--max-mb",type=int,default=32)
    a=p.parse_args()
    api=HfApi()
    try:
        revision=api.dataset_info(REPO,revision=a.revision).sha
        files=api.list_repo_files(REPO,repo_type="dataset",revision=revision)
        for group in groups(files):
            if {"labels","calibration","depth"}<=group.keys():
                break
        else:
            raise ValueError("No coherent individually accessible frame group. Archives require manual selection.")
        paths=list(group.values())
        if len(paths)>a.max_files:
            raise ValueError("Frame group exceeds file budget")
        print(revision,*paths,sep="\n")
        if list_only or a.list_only or a.dry_run:
            return
        entries=api.get_paths_info(REPO,paths,repo_type="dataset",revision=revision)
        if len(entries)!=len(paths) or sum(e.size for e in entries)>a.max_mb*1_000_000:
            raise ValueError("Unverified size or download exceeds budget")
        for path in paths:
            hf_hub_download(REPO,path,repo_type="dataset",revision=revision,local_dir=Path("data/card"))
    except (OSError,ValueError,RuntimeError) as exc:
        raise SystemExit(f"CARD unavailable ({type(exc).__name__}). Check the file budget, network, dataset access conditions and hf auth login.") from None


if __name__=="__main__":
    main()
