#!/usr/bin/env bash
set -euo pipefail
task=${1:?task required}
gpu=${2:?GPU required}
seed=${3:-3072}
[[ "$seed" =~ ^[0-9]+$ && "$gpu" =~ ^[0-3]$ ]] || exit 2
case "$task" in
  pusht) run="flow_pusht_seed${seed}_init3072" ;;
  reacher) run="flow_reacher_seed${seed}_init3073" ;;
  tworoom|cube) run="flow_${task}_seed${seed}" ;;
  *) exit 2 ;;
esac
export STABLEWM_HOME=/home/wangtz/.stable-wm
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export CUDA_VISIBLE_DEVICES="$gpu" HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python=/home/wangtz/anaconda3/envs/le-wm/bin/python
experiment="$STABLEWM_HOME/experiments/$run"
smoke="${run}_smoke"
log_dir="$STABLEWM_HOME/logs/flow-jepa"
for path in "$experiment" "$STABLEWM_HOME/checkpoints/$run" "$STABLEWM_HOME/checkpoints/$smoke" "$log_dir/${run}_train.log"; do
  [[ ! -e "$path" ]] || { echo "Refusing existing path: $path"; exit 2; }
done
[[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" ]] || {
  echo "GPU $gpu is occupied"; exit 2;
}
mkdir -p "$STABLEWM_HOME/experiments" "$log_dir"
mkdir "$experiment"
mkdir "$experiment/source"
cd /home/wangtz/Flow-JEPA
cp train.py jepa.py module.py utils.py eval.py eval_utils.py verify_flow_run.py run_flow_training.sh "$experiment/source/"
cp -a config perturbations "$experiment/source/"
git rev-parse HEAD > "$experiment/base_commit.txt"
git diff > "$experiment/tracked_changes.patch"
cd "$experiment/source"
find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$experiment/source_sha256.txt"
"$python" -m pip freeze > "$experiment/environment.txt"
trap 'code=$?; echo "PIPELINE_EXIT_CODE=$code $(date -Is)"' EXIT
echo "PROFILE_CHECK task=$task seed=$seed GPU=$gpu $(date -Is)"
"$python" verify_flow_run.py --task "$task" --seed "$seed" --output "$experiment/profile_check.json"

echo "SMOKE_START $smoke $(date -Is)"
"$python" -u train.py --config-name="paper_$task" "seed=$seed" \
  "subdir=$smoke" "output_model_name=$smoke" trainer.max_epochs=1 \
  +trainer.limit_train_batches=3 +trainer.limit_val_batches=1 +trainer.log_every_n_steps=1 \
  num_workers=2 2>&1 | tee "$experiment/smoke.log"
"$python" verify_flow_run.py --task "$task" --seed "$seed" --smoke-run "$smoke" --output "$experiment/smoke_check.json"
echo "SMOKE_COMPLETE $(date -Is)"

echo "TRAIN_START $run $(date -Is)"
set +e
"$python" -u train.py --config-name="paper_$task" "seed=$seed" \
  2>&1 | tee "$log_dir/${run}_train.log"
code=$?
set -e
echo "TRAIN_EXIT_CODE=$code $(date -Is)"
[[ "$code" == 0 ]] || exit "$code"
[[ -f "$STABLEWM_HOME/checkpoints/$run/weights_epoch_20.pt" ]]
echo "TRAIN_COMPLETE $run $(date -Is)"
