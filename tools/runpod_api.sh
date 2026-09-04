#!/usr/bin/env bash
# RunPod GraphQL helper for the C seed-1 replication pod runner.
# Usage: tools/runpod_api.sh '<graphql query or mutation>'
# Reads RUNPOD_API_KEY from ~/.config/mats/secrets.env (outside the repo); never prints it.
set -euo pipefail
set -a; source "${RUNPOD_SECRETS:-$HOME/.config/mats/secrets.env}"; set +a
python3 - "$1" <<'PY'
import json, os, sys, urllib.request, urllib.error
q = sys.argv[1]
req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q}).encode(),
                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"], "User-Agent": "curl/8.7.1", "Accept": "application/json"})
try:
    print(json.dumps(json.load(urllib.request.urlopen(req, timeout=60)), indent=1))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:2000])
PY
