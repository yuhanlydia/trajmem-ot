#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${ROBOMME_DIR:-${ROOT}/third_party/robomme_policy_learning}"
UPSTREAM_REPO="https://github.com/RoboMME/robomme_policy_learning.git"
UPSTREAM_COMMIT="ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${UPSTREAM_DIR}")"
  GIT_LFS_SKIP_SMUDGE=1 git clone "${UPSTREAM_REPO}" "${UPSTREAM_DIR}"
fi

if [[ -n "$(git -C "${UPSTREAM_DIR}" status --porcelain)" ]]; then
  echo "Refusing to change a dirty upstream checkout: ${UPSTREAM_DIR}" >&2
  exit 2
fi

git -C "${UPSTREAM_DIR}" fetch origin "${UPSTREAM_COMMIT}"
git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_COMMIT}"

cd "${UPSTREAM_DIR}"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install -e "${ROOT}"

if [[ "${ROBOMME_WITH_SIM:-0}" == "1" ]]; then
  git submodule update --init
fi

cat <<EOF
RoboMME runtime ready at ${UPSTREAM_DIR}
Pinned commit: ${UPSTREAM_COMMIT}
Run TrajMem-OT commands with:
  cd ${UPSTREAM_DIR}
  PYTHONPATH=${ROOT}/src uv run python ${ROOT}/scripts/run_e11_robomme_jvp.py --help
EOF
