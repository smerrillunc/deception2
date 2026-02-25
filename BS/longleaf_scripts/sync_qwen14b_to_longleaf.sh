#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"

longleaf_host="${LONGLEAF_HOST:-}"
remote_root="${REMOTE_ROOT:-/work/users/s/m/smerrill/deception2}"
dry_run="${DRY_RUN:-0}"
ssh_key_path="${SSH_KEY_PATH:-}"
ssh_identities_only="${SSH_IDENTITIES_ONLY:-0}"
ssh_strict_host_key_checking="${SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
use_ssh_mux="${USE_SSH_MUX:-1}"
ssh_control_persist="${SSH_CONTROL_PERSIST:-600}"
ssh_control_path="${SSH_CONTROL_PATH:-/tmp/longleaf_mux_%C}"
close_ssh_mux="${CLOSE_SSH_MUX:-0}"
rsync_delete="${RSYNC_DELETE:-0}"

if [[ -z "$longleaf_host" ]]; then
  echo "LONGLEAF_HOST is required."
  echo "Example:"
  echo "  LONGLEAF_HOST=smerrill@longleaf.unc.edu ./sync_qwen14b_to_longleaf.sh"
  exit 1
fi

if [[ -n "$ssh_key_path" && ! -f "$ssh_key_path" ]]; then
  echo "SSH_KEY_PATH does not exist: $ssh_key_path"
  exit 1
fi

bs_results_dir="${BS_RESULTS_DIR:-$project_root/BS/Results}"
if [[ -n "${GRIDWORLD_RESULTS_DIR:-}" ]]; then
  gridworld_results_dir="$GRIDWORLD_RESULTS_DIR"
elif [[ -d "$project_root/Gridworld/Results" ]]; then
  gridworld_results_dir="$project_root/Gridworld/Results"
elif [[ -d "$project_root/Gridwolrd/Results" ]]; then
  # Backward-compatible typo handling if user's local folder is misspelled.
  gridworld_results_dir="$project_root/Gridwolrd/Results"
else
  gridworld_results_dir="$project_root/Gridworld/Results"
fi

if [[ ! -d "$bs_results_dir" ]]; then
  echo "Missing directory: $bs_results_dir"
  exit 1
fi
if [[ ! -d "$gridworld_results_dir" ]]; then
  echo "Missing directory: $gridworld_results_dir"
  exit 1
fi

gridworld_parent_dir="$(basename "$(dirname "$gridworld_results_dir")")"
remote_bs_parent="${REMOTE_BS_PARENT_DIR:-$remote_root/BS}"
remote_gridworld_parent="${REMOTE_GRIDWORLD_PARENT_DIR:-$remote_root/$gridworld_parent_dir}"

rsync_flags=(-avh --progress)
if [[ "$dry_run" == "1" ]]; then
  rsync_flags+=(--dry-run)
fi
if [[ "$rsync_delete" == "1" ]]; then
  rsync_flags+=(--delete)
fi

ssh_opts=(-o "StrictHostKeyChecking=$ssh_strict_host_key_checking")
if [[ "$ssh_identities_only" == "1" ]]; then
  ssh_opts+=(-o IdentitiesOnly=yes)
fi
if [[ -n "$ssh_key_path" ]]; then
  ssh_opts+=(-i "$ssh_key_path")
fi
if [[ "$use_ssh_mux" == "1" ]]; then
  ssh_opts+=(
    -o ControlMaster=auto
    -o "ControlPersist=$ssh_control_persist"
    -o "ControlPath=$ssh_control_path"
  )
fi

rsync_ssh_cmd=(ssh "${ssh_opts[@]}")
rsync_ssh_cmd_str="${rsync_ssh_cmd[*]}"

run_rsync() {
  local src="$1"
  local dst="$2"
  echo "rsync $src -> $longleaf_host:$dst"
  rsync "${rsync_flags[@]}" -e "$rsync_ssh_cmd_str" "$src" "$longleaf_host:$dst"
}

echo "Local source directories:"
echo "  $bs_results_dir"
echo "  $gridworld_results_dir"
echo "Remote destination parents:"
echo "  $remote_bs_parent"
echo "  $remote_gridworld_parent"

if [[ "$use_ssh_mux" == "1" ]]; then
  echo "Establishing SSH master connection (single password prompt expected) ..."
  ssh "${ssh_opts[@]}" "$longleaf_host" "true"
fi

echo "Creating remote directories under $remote_root ..."
ssh "${ssh_opts[@]}" "$longleaf_host" "mkdir -p \
  '$remote_bs_parent' \
  '$remote_gridworld_parent'"

echo "Syncing BS Results directory ..."
run_rsync "$bs_results_dir" "$remote_bs_parent/"

echo "Syncing $gridworld_parent_dir Results directory ..."
run_rsync "$gridworld_results_dir" "$remote_gridworld_parent/"

if [[ "$use_ssh_mux" == "1" && "$close_ssh_mux" == "1" ]]; then
  echo "Closing SSH master connection ..."
  ssh "${ssh_opts[@]}" -O exit "$longleaf_host" >/dev/null 2>&1 || true
fi

echo "Sync complete."
echo "Remote root: $longleaf_host:$remote_root"
