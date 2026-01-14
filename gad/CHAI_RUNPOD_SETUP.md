# Manual RunPod Setup Steps


## Deploy RunPod

### 1. Configure Pod Settings

- **GPUs**: Use 8 x B100 GPUs
- **Docker Image**: `verlai/verl:vllm012.latest
`
- **Container Start Command**: 

    - `bash -c 'apt update; \
DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y; \
mkdir -p ~/.ssh; \
cd ~/.ssh; \
chmod 700 ~/.ssh; \
echo "$PUBLIC_KEY" >> authorized_keys; \
chmod 700 authorized_keys; \
service ssh start; \
sleep infinity'` 
    - https://docs.runpod.io/pods/configuration/use-ssh#full-ssh-via-public-ip-with-key-authentication
- **Volume Mount**: Mount `/tmp` for persisting checkpoint files
- **Volume disk**: 5TB to be safer
- **Notes**:
  - verl image is missing vim and tmux (can install if needed)
  - Image includes Python, PyTorch, and vLLM pre-installed

### 2. Connect via SSH

Once RunPod is ready, connect via SSH:
```bash
ssh root@<runpod-pod-ip>
```

### 3. [Preferred] Run Automated Setup

On **RunPod pod**, clone the repository and run setup:
```bash
cd /tmp
git clone https://github.com/AceHao/LMOps.git
cd LMOps/gad
bash scripts/setup_chai_training.sh
```

**The setup script will automatically:**
- ✅ Verify parquet files exist and are valid
- ✅ Install WANDB and prompt for login
- ✅ Configure WANDB environment variables
- ✅ Clone verl repository
- ✅ Install Python dependencies (via local_setup.sh)
- ✅ Download Qwen model from HuggingFace
- ✅ Verify all prerequisites are met
- ✅ Print training command to start

**Expected setup time:** 10-15 minutes (mostly model download).

### 4. [Skip] Manual Setup Steps

> **Note:** These steps are automated by `scripts/setup_chai_training.sh`.
> This document serves as reference for manual setup or troubleshooting.
> For the streamlined automated setup, see step 3.

---

If not using the automated setup script, run these commands manually:

```bash
# Install WANDB
pip install wandb

# Login to WANDB
wandb login <API_KEY>

# Set environment variables
export WANDB_PROJECT='your-actual-project-name'
export WANDB_API_KEY='your-actual-api-key'

# Clone repositories from Ace, where it has fixes for running on B100 gpus
bash local_docker.sh
cd /tmp
git clone https://github.com/AceHao/LMOps.git
cd /tmp/LMOps/gad
git clone https://github.com/AceHao/verl.git

# Install dependencies
bash local_setup.sh

# Download Qwen model
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /tmp/Qwen2.5-3B-Instruct
```

### 5. Transfer Chai data to pod 

### Step 1: Transfer Data Files

On your **local machine**, prepare to transfer parquet files.

On **RunPod pod** (after SSH), create data directory:
```bash
mkdir -p /tmp/LMOps/gad/chai_opus_data
```

From your **local machine**, SCP parquet files:
```bash
# Transfer training data (675MB)
scp -i ~/.ssh/id_chai -P 22075 chai_opus_data/transformed_chai_train.parquet root@69.30.85.114:/tmp/LMOps/gad/chai_opus_data/

# Transfer validation data (35MB)
scp -i ~/.ssh/id_chai -P 22075 chai_opus_data/transformed_chai_val.parquet root@69.30.85.114:/tmp/LMOps/gad/chai_opus_data/
```

**Expected transfer time:** 2-5 minutes depending on network speed.


See **[CHAI_TRAINING_SETUP.md](CHAI_TRAINING_SETUP.md)** to start training

---

## Additional Notes

### Installing Missing Tools

If you need vim or tmux:
```bash
apt-get update
apt-get install -y vim tmux
```

### Troubleshooting

**Issue**: Model download fails
**Solution**: Check HuggingFace authentication or network connection

**Issue**: Permission denied on `/tmp`
**Solution**: Verify volume mount configuration in RunPod settings

**Issue**: WANDB login fails
**Solution**: Get API key from https://wandb.ai/authorize and try again
