#!/usr/bin/env python3

from github import Github
import os

if __name__ == "__main__":
    git_token = os.getenv("FLEXFLOW_TOKEN") or ""
    if len(git_token) < 40:
        print("FLEXFLOW_TOKEN not set properly")
        exit(1)
    workflow_name = "gpu_ci"
    git_client = Github(git_token)
    if not git_client:
        print("Could not get a Git client")
        exit(1)
    repo = git_client.get_repo("flexflow/FlexFlow")
    if not repo:
        print("Could not access the FlexFlow repo")
        exit(1)
    workflows = [w for w in repo.get_workflows() if w.name == workflow_name]
    print(workflows)
    
