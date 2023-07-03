#!/usr/bin/env python3

from github import Github
import os, sys, argparse, time


def get_num_workflow_runs(repo, workflow_names, in_progress_only=False):
    workflows = [
        w for w in repo.get_workflows() for w_name in workflow_names if w.path == w_name
    ]
    if len(workflows) != len(workflow_names):
        print(
            f"Found {len(workflows)} workflows instead of {len(workflow_names)}. Weird."
        )
        sys.exit(1)
    count = 0
    for workflow in workflows:
        running_states = (
            ["in_progress"] if in_progress_only else ["queued", "in_progress"]
        )
        runs = [
            run for status in running_states for run in workflow.get_runs(status=status)
        ]
        count += len(runs)
    return count


if __name__ == "__main__":

    # Check who is running this script (the daemon or a regular gpu-ci runner)
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()

    # Log into the GitHub API and get a handle to the repo
    git_token = os.getenv("FLEXFLOW_TOKEN") or ""
    if len(git_token) < 40:
        print("FLEXFLOW_TOKEN not set properly")
        sys.exit(1)
    git_client = Github(git_token)
    if not git_client:
        print("Could not get a Git client")
        sys.exit(1)
    repo = git_client.get_repo("flexflow/FlexFlow")
    if not repo:
        print("Could not access the FlexFlow repo")
        sys.exit(1)

    if args.daemon:
        print("Running the daemon...")
        # Check if there is any `gpu-ci` workflow in progress or queued
        target_workflows = [
            ".github/workflows/gpu-ci.yml",
            ".github/workflows/multinode-test.yml",
        ]
        n = get_num_workflow_runs(repo, target_workflows, in_progress_only=False)
        print(f"Detected {n} GPU-related workflow runs in progress or queued")

        instance_id = os.getenv("FLEXFLOW_RUNNER_INSTANCE_ID") or ""
        if len(instance_id) != 19:
            print("FLEXFLOW_RUNNER_INSTANCE_ID not set properly")
            sys.exit(1)
        # If there are `gpu-ci` runs in progress or queued, turn on the `flexflow-runner` spot instance,
        # if it is not already on. If there are no `gpu-ci` runs in progress or queued, turn off
        # the spot instance if it is not already off.
        if n > 0:
            print("Starting the `flexflow-runner` spot instance (if not already on)...")
            os.system(
                f"aws ec2 start-instances --region us-east-2 --instance-ids {instance_id}"
            )
        else:
            print(
                "Stopping the `flexflow-runner` spot instance (if not already off)..."
            )
            os.system(
                f"aws ec2 stop-instances --region us-east-2 --instance-ids {instance_id}"
            )
    else:
        print("Waiting for the deamon to finish running...")
        # Wait until the daemon has finished running
        target_workflow = [".github/workflows/gpu-ci-daemon.yml"]
        n = get_num_workflow_runs(repo, target_workflow, in_progress_only=True)
        while n > 0:
            time.sleep(30)
            n = get_num_workflow_runs(repo, target_workflow, in_progress_only=True)
