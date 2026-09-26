"""
Remote training orchestrator: sets up SSH key auth, copies code+data to Mac, runs pipeline.
"""
import paramiko
import os
import sys
import time
from scp import SCPClient

# --- Remote config ---
HOST = os.environ.get("REMOTE_HOST", "100.94.104.70")
USER = os.environ.get("REMOTE_USER", "harikrishna")
PASS = os.environ.get("REMOTE_PASS", "")
REMOTE_DIR = "/tmp/entity_resolution"

LOCAL_PROJECT = os.path.dirname(os.path.abspath(__file__))

def get_ssh():
    """Create an SSH client connected to the Mac."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=30)
    return ssh

def run_remote(ssh, cmd, print_output=True):
    """Run a command on the remote machine and return stdout."""
    print(f"[REMOTE] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=3600)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    exit_code = stdout.channel.recv_exit_status()
    if print_output:
        if out.strip():
            print(out.strip())
        if err.strip():
            print(f"[STDERR] {err.strip()}")
    if exit_code != 0:
        print(f"[EXIT CODE] {exit_code}")
    return out, err, exit_code

def setup_ssh_key(ssh):
    """Copy local SSH public key to remote authorized_keys for future passwordless access."""
    pubkey_path = os.path.expanduser("~/.ssh/id_rsa.pub")
    if os.path.exists(pubkey_path):
        with open(pubkey_path) as f:
            pubkey = f.read().strip()
        run_remote(ssh, f'mkdir -p ~/.ssh && chmod 700 ~/.ssh')
        run_remote(ssh, f'echo "{pubkey}" >> ~/.ssh/authorized_keys && sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys')
        print("[OK] SSH key installed for future passwordless access")

def upload_files(ssh):
    """Upload project code and data to the Mac."""
    scp = SCPClient(ssh.get_transport(), progress4=lambda f, s, sent: None)
    
    # Create remote directory structure
    run_remote(ssh, f"mkdir -p {REMOTE_DIR}/src {REMOTE_DIR}/dataset/train {REMOTE_DIR}/dataset/test {REMOTE_DIR}/output {REMOTE_DIR}/models {REMOTE_DIR}/utils")
    
    # Upload source code
    src_dir = os.path.join(LOCAL_PROJECT, "src")
    for fname in os.listdir(src_dir):
        if fname.endswith('.py'):
            local_path = os.path.join(src_dir, fname)
            remote_path = f"{REMOTE_DIR}/src/{fname}"
            print(f"  Uploading {fname}...")
            scp.put(local_path, remote_path)
    
    # Upload requirements
    req_path = os.path.join(LOCAL_PROJECT, "requirements.txt")
    if os.path.exists(req_path):
        scp.put(req_path, f"{REMOTE_DIR}/requirements.txt")
    
    # Upload validator
    val_path = os.path.join(LOCAL_PROJECT, "utils", "validate_submission.py")
    if os.path.exists(val_path):
        scp.put(val_path, f"{REMOTE_DIR}/utils/validate_submission.py")
    
    # Upload data files (large - check if already there)
    data_files = [
        ("dataset/train/train_source1.tsv", "dataset/train/train_source1.tsv"),
        ("dataset/train/train_source2.tsv", "dataset/train/train_source2.tsv"),
        ("dataset/train/train_source3.tsv", "dataset/train/train_source3.tsv"),
        ("dataset/train/train_ground_truth.tsv", "dataset/train/train_ground_truth.tsv"),
        ("dataset/test/test_source1.tsv", "dataset/test/test_source1.tsv"),
        ("dataset/test/test_source2.tsv", "dataset/test/test_source2.tsv"),
        ("dataset/test/test_source3.tsv", "dataset/test/test_source3.tsv"),
    ]
    
    for local_rel, remote_rel in data_files:
        local_path = os.path.join(LOCAL_PROJECT, local_rel)
        remote_path = f"{REMOTE_DIR}/{remote_rel}"
        
        # Check if file already exists on remote
        out, _, _ = run_remote(ssh, f"wc -c {remote_path} 2>/dev/null || echo 0", print_output=False)
        remote_size = int(out.strip().split()[0]) if out.strip() else 0
        local_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        
        if remote_size > 0 and abs(remote_size - local_size) < 1000:
            print(f"  {local_rel}: already on remote ({remote_size:,} bytes), skipping")
        else:
            print(f"  Uploading {local_rel} ({local_size/1024/1024:.0f} MB)...")
            scp.put(local_path, remote_path)
            print(f"  Done")
    
    scp.close()

def download_results(ssh):
    """Download output files from the Mac back to local."""
    scp = SCPClient(ssh.get_transport())
    
    local_output = os.path.join(LOCAL_PROJECT, "output")
    os.makedirs(local_output, exist_ok=True)
    
    for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
        remote_path = f"{REMOTE_DIR}/output/{fname}"
        local_path = os.path.join(local_output, fname)
        print(f"  Downloading {fname}...")
        try:
            scp.get(remote_path, local_path)
            print(f"  Saved to {local_path}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Also download model
    local_models = os.path.join(LOCAL_PROJECT, "models")
    os.makedirs(local_models, exist_ok=True)
    for fname in ["lgbm_model.pkl", "lgbm_final.pkl"]:
        remote_path = f"{REMOTE_DIR}/models/{fname}"
        local_path = os.path.join(local_models, fname)
        try:
            scp.get(remote_path, local_path)
            print(f"  Downloaded {fname}")
        except Exception as e:
            print(f"  Model {fname}: {e}")
    
    scp.close()

def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    print("=" * 70)
    print("Remote Training Orchestrator")
    print(f"  Host: {USER}@{HOST}")
    print(f"  Remote dir: {REMOTE_DIR}")
    print("=" * 70)
    
    ssh = get_ssh()
    print("[OK] SSH connected\n")
    
    if step in ("all", "setup"):
        # 1. Setup SSH key for passwordless access
        setup_ssh_key(ssh)
        
        # 2. Check Python & install deps on remote
        print("\n=== Checking Remote Environment ===")
        run_remote(ssh, "python3 --version")
        run_remote(ssh, "uname -m")
        run_remote(ssh, "sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 \" GB RAM\"}'")
        
        print("\n=== Installing Dependencies on Remote ===")
        run_remote(ssh, "python3 -m pip install --quiet pandas scikit-learn lightgbm rapidfuzz python-Levenshtein scipy numpy 2>&1 | tail -5")
    
    if step in ("all", "upload"):
        # 3. Upload code and data
        print("\n=== Uploading Project Files ===")
        upload_files(ssh)
    
    if step in ("all", "train"):
        # 4. Run the pipeline on remote
        print("\n=== Running Pipeline on Remote ===")
        print("This will take a while with the full dataset...\n")
        
        cmd = f"cd {REMOTE_DIR} && PYTHONIOENCODING=utf-8 python3 -m src.pipeline full 2>&1"
        
        # Use exec_command with a long timeout for training
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=7200)  # 2 hour timeout
        
        # Stream output in real-time
        while True:
            line = stdout.readline()
            if not line:
                break
            print(line.rstrip())
        
        err = stderr.read().decode('utf-8', errors='replace')
        if err.strip():
            print(f"\n[STDERR]\n{err.strip()}")
        
        exit_code = stdout.channel.recv_exit_status()
        print(f"\n[Pipeline exit code: {exit_code}]")
    
    if step in ("all", "download"):
        # 5. Download results
        print("\n=== Downloading Results ===")
        download_results(ssh)
    
    if step in ("all", "validate"):
        # 6. Validate locally
        print("\n=== Validating Submission ===")
        run_remote(ssh, f"cd {REMOTE_DIR} && python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test")
    
    ssh.close()
    print("\n[DONE]")

if __name__ == "__main__":
    main()
