#!/usr/bin/env bash
set -euo pipefail
task=${1:?task required}
gpu=${2:?GPU required}
version=${3:-v2}
train_seed=${4:-3072}
[[ "$gpu" =~ ^[0-3]$ ]] || exit 2
[[ "$version" =~ ^v[0-9]+$ ]] || exit 2
[[ "$train_seed" =~ ^[0-9]+$ ]] || exit 2
case "$task" in
  pusht) run=flow_pusht_seed${train_seed}_init3072; dataset=pusht_expert_train; radius=10 ;;
  tworoom) run=flow_tworoom_seed${train_seed}; dataset=tworoom; radius=35 ;;
  reacher) run=flow_reacher_seed${train_seed}_init3073; dataset=reacher; radius=35 ;;
  cube) run=flow_cube_seed${train_seed}; dataset=cube_single_expert; radius=35 ;;
  *) exit 2 ;;
esac
export STABLEWM_HOME=/home/wangtz/.stable-wm
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export CUDA_VISIBLE_DEVICES="$gpu" HYDRA_FULL_ERROR=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
python=/home/wangtz/anaconda3/envs/le-wm/bin/python
root="$STABLEWM_HOME/evaluations/flow_${task}_train${train_seed}_epoch20_paper_${version}"
[[ ! -e "$root" ]] || { echo "Refusing existing output: $root"; exit 2; }
[[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" ]] || {
  echo "GPU $gpu is occupied"; exit 2;
}
[[ -f "$STABLEWM_HOME/checkpoints/$run/weights_epoch_20.pt" ]]
mkdir -p "$root/source"
cd /home/wangtz/Flow-JEPA
cp train.py jepa.py module.py utils.py eval.py eval_utils.py verify_flow_evaluation.py run_flow_evaluation.sh "$root/source/"
cp -a config perturbations "$root/source/"
git rev-parse HEAD > "$root/base_commit.txt"
git diff > "$root/tracked_changes.patch"
"$python" -m pip freeze > "$root/environment.txt"
cd "$root/source"
find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$root/source_sha256.txt"
trap 'code=$?; echo "EVAL_PIPELINE_EXIT_CODE=$code $(date -Is)"' EXIT
extra=()
[[ "$task" != reacher ]] || extra=(+reacher_grid_texuniform=false)

evaluate() {
  local name=$1 seed=$2 count=$3 noise=$4 std=$5 video=$6
  echo "EVAL_START $task $name $(date -Is)"
  "$python" -u eval.py --config-name="$task" \
    "policy=$STABLEWM_HOME/checkpoints/$run/weights_epoch_20.pt" \
    "eval.dataset_name=$STABLEWM_HOME/$dataset" "eval.num_eval=$count" \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    "seed=$seed" "flow_seed=$seed" "+environment_reset_seed=$seed" \
    solver.n_steps=10 solver.num_samples=300 solver.topk=30 solver.var_scale=1.0 solver.batch_size=1 \
    plan_config.horizon=5 plan_config.receding_horizon=5 plan_config.action_block=5 \
    +plan_config.history_len=1 +plan_config.warm_start=true \
    "perturbation.enabled=$noise" "perturbation.seed=$seed" \
    "perturbation.gaussian_noise.std=$std" "perturbation.gaussian_noise.radius=$radius" \
    "output.dir=$root/$name" "output.save_video=$video" "${extra[@]}" \
    2>&1 | tee "$root/${name}.log"
  [[ -f "$root/$name/metrics.json" ]]
  echo "EVAL_COMPLETE $task $name $(date -Is)"
}

evaluate smoke_clean 9001 2 false 100 true
evaluate smoke_zero 9001 2 true 0 false
evaluate smoke_noisy 9001 2 true 100 true
"$python" verify_flow_evaluation.py "$root" --smoke
echo "SMOKE_COMPLETE $task $(date -Is)"
for seed in 42 43 44; do
  video=false
  [[ "$seed" != 42 ]] || video=true
  evaluate "seed${seed}_clean" "$seed" 50 false 100 "$video"
  evaluate "seed${seed}_noisy" "$seed" 50 true 100 "$video"
  "$python" verify_flow_evaluation.py "$root" --seed "$seed"
done
echo "ALL_EVALUATIONS_COMPLETE $task $(date -Is)"
