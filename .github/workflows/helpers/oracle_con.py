import oci
import argparse
import os

parser = argparse.ArgumentParser(description="Program with optional flags")
group = parser.add_mutually_exclusive_group()
group.add_argument("--start", action="store_true", help="Start action")
group.add_argument("--stop", action="store_true", help="Stop action")
parser.add_argument("--instance_id", type=str, required=True, help="instance id required")
args = parser.parse_args()

oci_key_content = os.getenv("OCI_CLI_KEY_CONTENT")

config = {
    "user": os.getenv("OCI_CLI_USER"),
    "key_content": os.getenv("OCI_CLI_KEY_CONTENT"),
    "fingerprint": os.getenv("OCI_CLI_FINGERPRINT"),
    "tenancy": os.getenv("OCI_CLI_TENANCY"),
    "region": os.getenv("OCI_CLI_REGION")
}

# Initialize the OCI configuration
oci.config.validate_config(config)

# Initialize the ComputeClient to interact with VM instances
compute = oci.core.ComputeClient(config)

# Replace 'your_instance_id' with the actual instance ID of your VM
instance_id = args.instance_id

# Perform the action
if args.start:
    # Start the VM
    compute.instance_action(instance_id, "START")
else:
    # Stop the VM
    compute.instance_action(instance_id, "STOP")
