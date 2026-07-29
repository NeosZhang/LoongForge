#!/bin/bash
# Embodied regression manual trigger (runs inside the container).
#
# This wrapper only sources config/env.sh — all parameter parsing, validation,
# defaults, and optional --prepare execution live in cli/manual.py.
#
# Both env-var and CLI-flag forms are supported (both handled by manual.py):
#   chip=A800 model_names="pi05 groot_n1_6" bash cli/manual.sh
#   bash cli/manual.sh --chip A800 --models pi05 groot_n1_6 --fail_fast
#
# Calling cli/manual.py directly also works; env.sh will be auto-loaded on first use.
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Source env.sh so path variables (EMBODIED_CI_ROOT / LOCAL_VLA_ARTIFACTS_ROOT / ...)
# propagate to both manual.py and the training subprocesses it launches.
source "${SCRIPT_DIR}/../config/env.sh"

exec python3 "${SCRIPT_DIR}/manual.py" "$@"
