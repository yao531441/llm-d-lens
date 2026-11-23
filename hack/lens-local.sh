#!/usr/bin/env bash
# Local source installation and service manager for Lens.
#
# Checks the environment and starts/stops the two local services:
#   - Vite frontend       (FRONTEND_PORT,   default 5173)
#   - FastAPI app/API     (BACKEND_PORT, default 3000; SERVER_PORT is an alias)
#
# All ports are configurable via environment variables (see below), and the
# frontend/backend proxies read the same variables so everything stays in sync.
#
# Usage:
#   hack/lens-local.sh check     # validate the environment without starting anything
#   hack/lens-local.sh start     # start all services (background, logs in .dev-logs/)
#   hack/lens-local.sh stop      # stop all services started by this script
#   hack/lens-local.sh restart [all|frontend|server|backend] # validate, then restart selected services
#   hack/lens-local.sh status    # show which services are running
#   hack/lens-local.sh clean [-y|--yes] [--cache|--hard|--all]
#                                    # stop services, delete app state; optionally
#                                    # caches, the dev venv/toolchain, or everything
#   hack/lens-local.sh install-node  # install a supported Node.js into .dev-tools/
#
# When the system Node.js is missing or older than NODE_MIN_MAJOR, 'start'
# downloads an official Node.js build into .dev-tools/ (checksum-verified) and
# uses it for this script's services only. Set NODE_AUTO_INSTALL=0 to opt out.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_DIR="$PWD"
cd "$ROOT_DIR"

# Optional local secrets are intentionally ignored by Git.
if [[ -f "$ROOT_DIR/.secrets.env" ]]; then
    set -a
    source "$ROOT_DIR/.secrets.env"
    set +a
fi

# ---- Configuration (override via environment) -------------------------------
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
SERVER_PORT="${SERVER_PORT:-3000}"
BACKEND_PORT="${BACKEND_PORT:-$SERVER_PORT}"
VITE_HOST="${VITE_HOST:-0.0.0.0}"

# HTTPS is on by default: browsers only expose features like microphone
# access (getUserMedia) in a "secure context" (HTTPS or http://localhost), so
# a box reached over http://<lan-ip> needs HTTPS to unlock those features.
# A self-signed cert is generated automatically (idempotent -- reused across
# restarts) since there's usually no public DNS name for a Let's Encrypt cert
# on an internal box. Set ENABLE_HTTPS=0 to opt out and run plain HTTP.
ENABLE_HTTPS="${ENABLE_HTTPS:-1}"
TLS_DIR="${TLS_DIR:-$ROOT_DIR/.dev-tls}"
TLS_HOST="${TLS_HOST:-}"

# Minimum Node.js major version.
NODE_MIN_MAJOR="${NODE_MIN_MAJOR:-22}"

# Set NODE_AUTO_INSTALL=0 to disable downloading a local Node.js toolchain when
# the system one is missing or too old.
NODE_AUTO_INSTALL="${NODE_AUTO_INSTALL:-1}"

# Pin an exact version (e.g. v22.23.2) to skip the release-index lookup.
NODE_INSTALL_VERSION="${NODE_INSTALL_VERSION:-}"

# Used when nodejs.org/dist/index.json cannot be reached.
NODE_FALLBACK_VERSION="${NODE_FALLBACK_VERSION:-v22.23.2}"

NODE_DIST_URL="${NODE_DIST_URL:-https://nodejs.org/dist}"

# Node.js installed by this script lives here, never outside the repository.
TOOLS_DIR="$ROOT_DIR/.dev-tools"
LOCAL_NODE_DIR="$TOOLS_DIR/node"

# Minimum Rust toolchain version (warn + upgrade hint when lower).
RUST_MIN_VERSION="${RUST_MIN_VERSION:-1.80.0}"

# Shared persistent data, disposable cache and diagnostic log roots.
source "$ROOT_DIR/hack/storage-env.sh"
EVALUATE_STORE="$LENS_DATA_DIR/metadata/evaluations"

# Optional IP address or DNS name advertised for deployment Model APIs. When
# empty, port-forwards remain localhost-only.
PRISM_MODEL_API_PUBLIC_HOST="${PRISM_MODEL_API_PUBLIC_HOST:-}"



# Optional comma-separated PCI allowlist restricting which XPUs/GPUs can be
# selected for deployment (read by the FastAPI backend).
PRISM_GPU_PCI_ALLOWLIST="${PRISM_GPU_PCI_ALLOWLIST:-}"

# Shared secret for authenticated Agentic/MCP calls (design section 8.3).
LENS_INTERNAL_AUTH_SECRET="${LENS_INTERNAL_AUTH_SECRET:-dev-internal-secret}"

# Deploy/Model Cache render Guide manifests from an llm-d checkout containing
# guides/. Default to a well-known location under $HOME so this works without
# manual setup; clone it there (or set LLM_D_ROOT) if it's missing.
LLM_D_ROOT="${LLM_D_ROOT:-$LENS_CACHE_DIR/llm-d}"
export LLM_D_ROOT

PID_DIR="$ROOT_DIR/.dev-pids"
LOG_DIR="$LENS_LOG_DIR/dev"

SERVICES=(frontend backend)

# ---- Helpers -----------------------------------------------------------------
log()  { printf '\033[1;36m[lens-local]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[lens-local]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[lens-local]\033[0m %s\n' "$*" >&2; }

prepare_target_node() {
    [[ -n "${TARGET_NODE:-}" ]] || return 0
    local node_pattern='^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$'
    if [[ ${#TARGET_NODE} -gt 253 || ! "$TARGET_NODE" =~ $node_pattern ]]; then
        fail "TARGET_NODE must be a valid Kubernetes node name."
        return 1
    fi
    local kubectl_path node_json
    if ! kubectl_path="$(command -v "${LLM_D_BENCH_KUBECTL_PATH:-kubectl}")" || [[ ! -x "$kubectl_path" ]]; then
        fail "Target-node checks require an executable kubectl on PATH or at LLM_D_BENCH_KUBECTL_PATH."
        return 1
    fi
    if ! command -v jq >/dev/null 2>&1; then
        fail "Target-node checks require jq."
        return 1
    fi
    if [[ ! -d "$LLM_D_ROOT/guides" ]]; then
        fail "LLM_D_ROOT must point to an llm-d checkout containing guides/."
        return 1
    fi
    if ! node_json="$("$kubectl_path" --request-timeout=30s get node "$TARGET_NODE" -o json)"; then
        fail "Kubernetes node '$TARGET_NODE' is not accessible in the selected context."
        return 1
    fi
    if ! jq -e --arg node "$TARGET_NODE" \
        '.metadata.name == $node and any(.status.conditions[]?; .type == "Ready" and .status == "True") and .spec.unschedulable != true' \
        <<<"$node_json" >/dev/null; then
        fail "Node '$TARGET_NODE' must be Ready and not cordoned."
        return 1
    fi
    log "Target node: $TARGET_NODE (Ready and not cordoned)."
    if [[ -n "$PRISM_GPU_PCI_ALLOWLIST" ]]; then
        log "Explicit GPU PCI allowlist: $PRISM_GPU_PCI_ALLOWLIST"
    fi
}

# run_with_tail <window> <command...> — runs <command>, showing only the
# last <window> lines of its (often very verbose) combined stdout/stderr on
# screen, redrawn in place, instead of flooding the terminal with
# everything. On failure, the full output is dumped so the real error is
# still visible. Falls back to running the command directly when stdout
# isn't a terminal (e.g. CI logs), where the full output is actually useful.
run_with_tail() {
    local window="$1"
    shift

    if [[ ! -t 1 ]]; then
        "$@"
        return $?
    fi

    local log_file
    log_file="$(mktemp)"
    "$@" >"$log_file" 2>&1 &
    local cmd_pid=$! prev_lines=0 lines cols

    while kill -0 "$cmd_pid" 2>/dev/null; do
        [[ "$prev_lines" -gt 0 ]] && printf '\033[%dA\033[J' "$prev_lines"
        # Truncate to the terminal width (minus the 2-space indent) so every
        # displayed line occupies exactly one terminal row -- otherwise long
        # lines (pip's output is full of long URLs/filenames) wrap onto
        # multiple rows, wc -l undercounts them, and the next redraw's
        # cursor-up doesn't go up far enough, leaving stale lines behind and
        # making the output appear to endlessly scroll instead of staying
        # pinned to a 5-line window. Re-measured every iteration in case the
        # terminal is resized mid-install.
        cols="$(tput cols 2>/dev/null || echo 80)"
        lines="$(tail -n "$window" "$log_file" 2>/dev/null | cut -c "1-$((cols > 2 ? cols - 2 : cols))")"
        if [[ -n "$lines" ]]; then
            printf '%s\n' "$lines" | sed 's/^/  /'
            prev_lines="$(wc -l <<<"$lines")"
        else
            prev_lines=0
        fi
        sleep 0.2
    done
    [[ "$prev_lines" -gt 0 ]] && printf '\033[%dA\033[J' "$prev_lines"

    wait "$cmd_pid"
    local rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail "command failed (exit $rc) — full output:"
        cat "$log_file" >&2
    fi
    rm -f "$log_file"
    return "$rc"
}

pid_file() { printf '%s/%s.pid' "$PID_DIR" "$1"; }

is_running() {
    local pid
    pid="$(cat "$(pid_file "$1")" 2>/dev/null || echo '')"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

port_in_use() {
    local port="$1"
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}\$"
}

# resolve_free_port <service> <requested-port> — return the requested port when
# it is free (or already owned by a running managed service), otherwise the next
# free port above it.
resolve_free_port() {
    local name="$1" port="$2"
    if is_running "$name"; then
        printf '%s\n' "$port"
        return 0
    fi
    while port_in_use "$port"; do
        port=$((port + 1))
    done
    printf '%s\n' "$port"
}

# ver_lt <v1> <v2> — returns 0 when v1 is lower than v2 (via sort -V).
ver_lt() {
    [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$1" ]] && [[ "$1" != "$2" ]]
}

# ---- Node.js toolchain -------------------------------------------------------
# node_major — major version of the node on PATH, or empty when unavailable.
node_major() {
    command -v node >/dev/null 2>&1 || return 0
    node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1
}

node_is_supported() {
    local major
    major="$(node_major)"
    [[ -n "$major" ]] && [[ "$major" -ge "$NODE_MIN_MAJOR" ]]
}

# activate_local_node — put a previously downloaded toolchain on PATH. Never
# downloads anything, so it is safe to call from read-only commands.
activate_local_node() {
    [[ -x "$LOCAL_NODE_DIR/bin/node" ]] || return 1
    case ":$PATH:" in
        *":$LOCAL_NODE_DIR/bin:"*) ;;
        *) PATH="$LOCAL_NODE_DIR/bin:$PATH"; export PATH ;;
    esac
    return 0
}

# node_platform — Node.js release suffix for this machine, empty when the
# platform has no official prebuilt binary.
node_platform() {
    local os arch
    case "$(uname -s)" in
        Linux)  os="linux" ;;
        Darwin) os="darwin" ;;
        *)      return 1 ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64)  arch="x64" ;;
        aarch64|arm64) arch="arm64" ;;
        *)             return 1 ;;
    esac
    printf '%s-%s\n' "$os" "$arch"
}

# resolve_node_version — newest release matching NODE_MIN_MAJOR, preferring LTS.
resolve_node_version() {
    if [[ -n "$NODE_INSTALL_VERSION" ]]; then
        printf '%s\n' "$NODE_INSTALL_VERSION"
        return 0
    fi
    local index
    index="$(curl --proto '=https' --tlsv1.2 -sSf -m 20 "$NODE_DIST_URL/index.json" 2>/dev/null || true)"
    if [[ -n "$index" ]] && command -v python3 >/dev/null 2>&1; then
        local resolved
        resolved="$(printf '%s' "$index" | python3 -c "
import json, sys
major = int(sys.argv[1])
try:
    releases = json.load(sys.stdin)
except ValueError:
    sys.exit(1)
matching = [r for r in releases if r['version'].lstrip('v').split('.')[0].isdigit()
            and int(r['version'].lstrip('v').split('.')[0]) == major]
lts = [r for r in matching if r.get('lts')]
chosen = (lts or matching)
if not chosen:
    sys.exit(1)
print(chosen[0]['version'])
" "$NODE_MIN_MAJOR" 2>/dev/null || true)"
        if [[ -n "$resolved" ]]; then
            printf '%s\n' "$resolved"
            return 0
        fi
    fi
    printf '%s\n' "$NODE_FALLBACK_VERSION"
}

node_manual_hint() {
    warn "  Install Node.js >= $NODE_MIN_MAJOR.x manually, for example:"
    echo "    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash"
    echo '    export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"'
    echo "    nvm install $NODE_MIN_MAJOR && nvm use $NODE_MIN_MAJOR"
}

# install_node — download an official Node.js build into .dev-tools and put it
# on PATH. The tarball is verified against the signed SHASUMS256.txt listing.
install_node() {
    local platform version tarball url workdir
    if ! platform="$(node_platform)"; then
        fail "no official Node.js binary for $(uname -s)/$(uname -m)"
        node_manual_hint
        return 1
    fi
    for tool in curl tar; do
        command -v "$tool" >/dev/null 2>&1 || { fail "$tool is required to install Node.js"; return 1; }
    done
    command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || {
        fail "sha256sum or shasum is required to verify the Node.js download"
        return 1
    }

    version="$(resolve_node_version)"
    tarball="node-$version-$platform.tar.xz"
    url="$NODE_DIST_URL/$version/$tarball"
    log "installing Node.js $version into $LOCAL_NODE_DIR ..."

    mkdir -p "$TOOLS_DIR"
    workdir="$(mktemp -d "$TOOLS_DIR/node-download.XXXXXX")"
    # shellcheck disable=SC2064
    trap "rm -rf '$workdir'" RETURN

    if ! curl --proto '=https' --tlsv1.2 -fL --progress-bar -m 600 -o "$workdir/$tarball" "$url"; then
        fail "failed to download $url"
        node_manual_hint
        return 1
    fi
    if ! curl --proto '=https' --tlsv1.2 -fL -sS -m 60 -o "$workdir/SHASUMS256.txt" "$NODE_DIST_URL/$version/SHASUMS256.txt"; then
        fail "failed to download the checksum list for $version"
        return 1
    fi

    local expected actual
    expected="$(awk -v file="$tarball" '$2 == file {print $1}' "$workdir/SHASUMS256.txt")"
    if [[ -z "$expected" ]]; then
        fail "$tarball is not listed in SHASUMS256.txt"
        return 1
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$workdir/$tarball" | awk '{print $1}')"
    else
        actual="$(shasum -a 256 "$workdir/$tarball" | awk '{print $1}')"
    fi
    if [[ "$expected" != "$actual" ]]; then
        fail "checksum mismatch for $tarball (expected $expected, got $actual)"
        return 1
    fi

    if ! tar -xJf "$workdir/$tarball" -C "$workdir"; then
        fail "failed to extract $tarball"
        return 1
    fi

    local target="$TOOLS_DIR/node-$version-$platform"
    rm -rf "$target"
    mv "$workdir/node-$version-$platform" "$target"
    ln -sfn "$target" "$LOCAL_NODE_DIR"

    activate_local_node || { fail "installed Node.js is not executable at $LOCAL_NODE_DIR/bin/node"; return 1; }
    log "Node.js ready: $(node --version) (npm $(npm --version 2>/dev/null || echo '?'))"
    warn "this toolchain is local to the repository; other shells still use the system Node.js"
    return 0
}

# ensure_node — guarantee a supported Node.js is on PATH for this script run.
ensure_node() {
    if node_is_supported; then
        return 0
    fi

    local previous
    previous="$(node_major)"

    if activate_local_node && node_is_supported; then
        log "using the repository-local Node.js $(node --version)"
    else
        if [[ "$NODE_AUTO_INSTALL" != "1" ]]; then
            fail "node $(node --version 2>/dev/null || echo 'MISSING') is older than the required $NODE_MIN_MAJOR.x (NODE_AUTO_INSTALL=0)"
            node_manual_hint
            return 1
        fi
        if [[ -n "$previous" ]]; then
            warn "node v$previous.x is too old (required >= $NODE_MIN_MAJOR.x) — installing a local toolchain"
        else
            warn "node is missing (required >= $NODE_MIN_MAJOR.x) — installing a local toolchain"
        fi
        install_node || return 1
    fi

    # Native addons are compiled against a specific Node ABI, so a major-version
    # switch invalidates whatever is already in node_modules.
    if [[ -n "$previous" && -d node_modules ]]; then
        warn "Node.js major changed v$previous -> $(node --version) — run 'npm install' if dependencies misbehave"
    fi
    return 0
}

# ---- node_modules setup -------------------------------------------------------
# ensure_node_modules — install npm dependencies when node_modules is missing
# OR stale relative to package-lock.json (so a `git pull` that adds/bumps a
# dependency, e.g. react-markdown, actually gets installed instead of being
# silently skipped just because a node_modules/ directory happens to exist).
ensure_node_modules() {
    local lockfile="package-lock.json"
    local stamp="node_modules/.install-stamp"
    local current_hash=""
    if [[ -f "$lockfile" ]]; then
        current_hash="$(sha256sum "$lockfile" 2>/dev/null | awk '{print $1}')"
    fi

    if [[ -d node_modules && -f "$stamp" && "$(cat "$stamp" 2>/dev/null)" == "$current_hash" ]]; then
        return 0
    fi

    if [[ -d node_modules ]]; then
        log "package-lock.json changed since the last install — running 'npm install'..."
    else
        log "node_modules missing — running 'npm install'..."
    fi
    # --ignore-scripts: onnxruntime-node's postinstall tries to download a
    # native binary that's unreachable on some networks; harmless to skip
    # since the browser build (transformers.js) never uses it anyway.
    npm install --ignore-scripts || {
        fail "npm install failed — see output above"
        return 1
    }
    echo "$current_hash" > "$stamp"
}

# ---- venv setup ---------------------------------------------------------------
# ensure_venv — create the virtualenv and install Python deps when missing, or
# when apps/backend/pyproject.toml changed since the last install. The stamp mirrors
# ensure_node_modules: a `git pull` that adds/bumps a dependency (e.g. casbin)
# must not be silently skipped just because the venv exists and the probed
# imports (llm_d_lens/uvicorn/pgserver) still work.
ensure_venv() {
    local stamp=".venv/.install-stamp"
    local current_hash=""
    if [[ -f apps/backend/pyproject.toml ]]; then
        current_hash="$(sha256sum apps/backend/pyproject.toml 2>/dev/null | awk '{print $1}')"
    fi

    if [[ -x .venv/bin/python && -f "$stamp" && "$(cat "$stamp" 2>/dev/null)" == "$current_hash" ]] \
        && .venv/bin/python -c "import llm_d_lens, uvicorn, pgserver" 2>/dev/null; then
        return 0
    fi

    if [[ ! -x .venv/bin/python ]]; then
        if ! command -v python3 >/dev/null 2>&1; then
            fail "python3 not found — cannot create the virtualenv"
            return 1
        fi
        log "creating virtualenv (.venv)..."
        python3 -m venv .venv || {
            fail "python3 -m venv failed — install python3-venv / ensurepip and retry"
            return 1
        }
    elif [[ -f "$stamp" && "$(cat "$stamp" 2>/dev/null)" != "$current_hash" ]]; then
        log "pyproject.toml changed since the last install — refreshing Python dependencies..."
    fi

    # '[embedded-db]' pulls in pgserver, needed by ensure_database's default
    # built-in PostgreSQL fallback below. '[ldap]' pulls in ldap3 so the
    # Administration directory provider "Test" action works in local dev.
    log "installing Python dependencies (pip install -e apps/backend[embedded-db,ldap])..."
    .venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1 || true
    run_with_tail 5 .venv/bin/python -m pip install -e "apps/backend[embedded-db,ldap]" || {
        fail "pip install -e apps/backend[embedded-db,ldap] failed — see output above"
        return 1
    }
    echo "$current_hash" > "$stamp"
}

# ---- database setup -------------------------------------------------------------
# ensure_database — auto-configure the built-in embedded PostgreSQL (no
# password, local dev only) the first time lens-local.sh runs with no database
# already configured (via env vars or a previous "Step 0" -- see
# llm_d_lens/db/system_router.py's get_database_status()). Unlike the
# online installer (scripts/LensInstaller-Ubuntu-x86_64-online.sh), which always requires
# a password on its embedded database, lens-local.sh intentionally stays
# password-less for local-dev convenience.
ensure_database() {
    if ! .venv/bin/python -m llm_d_lens.db.bootstrap_cli --mode embedded; then
        fail "database auto-configuration failed — see output above"
        return 1
    fi
}

# ---- check -------------------------------------------------------------------
cmd_check() {
    local ok=1 target="${1:-all}"
    log "Checking environment..."

    # Pick up a toolchain a previous run installed, without downloading here.
    activate_local_node >/dev/null 2>&1 || true

    if command -v node >/dev/null 2>&1; then
        local node_major
        node_major="$(node --version | sed 's/^v//' | cut -d. -f1)"
        if [[ "$node_major" -ge "$NODE_MIN_MAJOR" ]]; then
            log "  node:     $(node --version)"
        else
            fail "  node:     $(node --version) (required >= $NODE_MIN_MAJOR.x)"
            warn "  Run 'hack/lens-local.sh install-node' (or 'start') to install it locally."
            ok=0
        fi
    else
        fail "  node:     MISSING (required >= $NODE_MIN_MAJOR.x)"
        warn "  Run 'hack/lens-local.sh install-node' (or 'start') to install it locally."
        ok=0
    fi
    command -v npm  >/dev/null 2>&1 && log "  npm:      $(npm --version)"  || { fail "  npm:      MISSING"; ok=0; }
    if [[ "$target" != frontend ]]; then
    if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
        local rust_ver
        rust_ver="$(rustc --version | awk '{print $2}')"
        if ver_lt "$rust_ver" "$RUST_MIN_VERSION"; then
            fail "  rust:     rustc $rust_ver (required >= $RUST_MIN_VERSION)"
            ok=0
            warn "  Upgrade Rust:"
            echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
            echo '    source "$HOME/.cargo/env"'
            echo '    rustup default stable'
            echo '    rustc --version'
        else
            log "  rust:     rustc $rust_ver"
        fi
        log "  cargo:    $(cargo --version)"
    else
        fail "  rust:     rustc/cargo missing"
        ok=0
        warn "  Install Rust:"
        echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo '    source "$HOME/.cargo/env"'
        echo '    rustup default stable'
        echo '    rustc --version'
    fi
    fi
    command -v ss   >/dev/null 2>&1 || warn "  ss:       missing (port checks skipped)"

    [[ -d node_modules ]] && log "  node_modules: present" || { fail "  node_modules: missing — run 'npm install'"; ok=0; }

    if [[ "$target" != frontend ]]; then
    if [[ -x .venv/bin/python ]]; then
        log "  venv:     $(.venv/bin/python --version 2>&1)"
        .venv/bin/python -c "import llm_d_lens, uvicorn" 2>/dev/null \
            && log "  python deps: llm_d_lens + uvicorn importable" \
            || { fail "  python deps: import failed — run 'pip install -e apps/backend'"; ok=0; }
    else
        fail "  venv:     missing (.venv/bin/python) — 'hack/lens-local.sh start' will create it"
        ok=0
    fi

    fi
    if [[ -d "$EVALUATE_STORE" ]]; then
        log "  evaluate store: $EVALUATE_STORE"
    else
        warn "  evaluate store: $EVALUATE_STORE does not exist (will be created at runtime)"
    fi

    if [[ -d "$LLM_D_ROOT/guides" ]]; then
        log "  LLM_D_ROOT: $LLM_D_ROOT"
    else
        warn "  LLM_D_ROOT: $LLM_D_ROOT (missing guides/ — clone https://github.com/llm-d/llm-d there, or set LLM_D_ROOT, before using Deploy/Guide features)"
    fi

    log "  configured ports:"
    for svc in "${SERVICES[@]}"; do
        local port=""
        case "$svc" in
            frontend)   port="$FRONTEND_PORT";;
            server)     port="$SERVER_PORT";;
            backend)    port="$BACKEND_PORT";;
        esac
        if port_in_use "$port"; then
            warn "    $svc -> $port (IN USE)"
        else
            log "    $svc -> $port (free)"
        fi
    done

    if [[ "$ok" -ne 1 ]]; then
        fail "Environment check failed — resolve the issues above and re-run."
        exit 1
    fi
    log "Environment OK."
}

# ---- start -------------------------------------------------------------------

# Ensures a self-signed TLS cert exists (generating one on first run, reusing
# it after) and exports TLS_CERT_FILE/TLS_KEY_FILE for the server + frontend
# to pick up. No-op when ENABLE_HTTPS=0.
ensure_tls() {
    [[ "$ENABLE_HTTPS" == "1" ]] || return 0
    local host="$TLS_HOST"
    if [[ -z "$host" ]]; then
        host="$(hostname -I 2>/dev/null | awk '{print $1}')"
    fi
    if [[ -z "$host" ]]; then
        warn "could not detect a LAN IP for the TLS certificate; falling back to plain HTTP."
        warn "set TLS_HOST=<your-ip> to enable HTTPS, or ENABLE_HTTPS=0 to silence this."
        ENABLE_HTTPS=0
        return 0
    fi
    local cert_env
    if ! cert_env="$("$ROOT_DIR/hack/generate-self-signed-cert.sh" "$host" "$TLS_DIR")"; then
        warn "failed to generate a self-signed TLS certificate; falling back to plain HTTP."
        ENABLE_HTTPS=0
        return 0
    fi
    eval "$cert_env"
    export TLS_CERT_FILE TLS_KEY_FILE
}

start_one() {
    local name="$1"
    shift
    if is_running "$name"; then
        warn "$name already running (pid $(cat "$(pid_file "$name")"))"
        return 0
    fi
    mkdir -p "$PID_DIR" "$LOG_DIR"
    log "starting $name..."
    local run_log="$name-$(date -u +%Y%m%dT%H%M%S%NZ)-$$.log"
    ln -sfn "$run_log" "$LOG_DIR/$name.log"
    nohup "$@" > "$LOG_DIR/$run_log" 2>&1 &
    echo "$!" > "$(pid_file "$name")"
    case "$name" in
        frontend) printf '%s\n' "$FRONTEND_PORT" > "$PID_DIR/$name.port" ;;
        server) printf '%s\n' "$SERVER_PORT" > "$PID_DIR/$name.port" ;;
        backend) printf '%s\n' "$BACKEND_PORT" > "$PID_DIR/$name.port" ;;
    esac
}

wait_for_http() {
    local name="$1" url="$2" attempts="${3:-30}"
    local i=0
    while [[ $i -lt $attempts ]]; do
        if curl -sf -k -m 2 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    warn "$name did not become ready at $url — see $LOG_DIR/$name.log"
    return 1
}

# ---- MCP tool catalog ---------------------------------------------------------
# Validate the Python MCP catalog before stopping any running service.
regenerate_mcp_tools() {
    if [[ ! -x .venv/bin/python ]]; then
        return 0
    fi
    mkdir -p "$LOG_DIR"
    log "validating MCP tool catalog from the backend API..."
    if ! .venv/bin/python -c 'from llm_d_lens.api.main import app; from llm_d_lens.assistant_tools.catalog import catalog; print(f"Validated {len(catalog(app))} MCP tools")' >"$LOG_DIR/generate-mcp-tools.log" 2>&1; then
        fail "MCP tool catalog validation failed; see $LOG_DIR/generate-mcp-tools.log"
        return 1
    fi
}

# Keep the reuse preflight ahead of cmd_stop on restart: a broken registration
# must not take down the existing development session. Pending refactor decisions
# remain blocking for the development check/CI, but not for local service startup.
prepare_start() {
    prepare_target_node || return $?
    ensure_node || exit 1
    ensure_node_modules || exit 1
    if [[ "${1:-all}" != frontend ]]; then ensure_venv || return 1; fi
    if [[ "${1:-all}" != frontend ]]; then regenerate_mcp_tools || return 1; fi
    log "updating reusable capability map..."
    local reuse_status=0
    node hack/reuse/cli.mjs map || reuse_status=$?
    if [[ "$reuse_status" -ne 0 ]]; then
        fail "Reuse map update failed. Ask the coding AI to fix .reuse/catalog.json; services were not stopped."
        return "$reuse_status"
    fi
    log "validating reuse registrations and pending decisions..."
    node hack/reuse/cli.mjs check --purpose startup || reuse_status=$?
    if [[ "$reuse_status" -eq 2 ]]; then
        warn "Pending refactor decisions require review; continuing local startup. See the pending decisions above."
        return 0
    fi
    if [[ "$reuse_status" -ne 0 ]]; then
        fail "Reuse check blocked startup (exit $reuse_status). Review the details above; services were not stopped."
        return "$reuse_status"
    fi
}

# Point operators to the protected first-run credential file without printing
# its contents. The file is removed once the password is changed.
print_initial_admin_credentials() {
    local cred_file="$LENS_DATA_DIR/credentials/initial_admin.txt"
    [[ -f "$cred_file" ]] || return 0
    log "administrator credentials are stored in the mode-600 file $cred_file; removed after the password is changed"
}

# Load the last ports written by start_one. Restart/start then keeps the same
# backend port when it is still free; an occupied port is shifted by
# resolve_free_port and the new value is written back to $PID_DIR/<svc>.port.
load_saved_ports() {
    local svc saved_port
    for svc in "${SERVICES[@]}"; do
        saved_port="$(cat "$PID_DIR/$svc.port" 2>/dev/null || true)"
        [[ "$saved_port" =~ ^[0-9]+$ ]] || continue
        (( 10#$saved_port >= 1 && 10#$saved_port <= 65535 )) || continue
        case "$svc" in
            frontend) FRONTEND_PORT="$saved_port" ;;
            server) SERVER_PORT="$saved_port" ;;
            backend) BACKEND_PORT="$saved_port" ;;
        esac
    done
}

cmd_start() {
    prepare_start || return $?
    load_saved_ports
    start_services
}

cmd_restart() {
    local target="${1:-all}"
    case "$target" in all|frontend|server|backend) ;; *) fail "Unknown service: $target"; return 1 ;; esac
    if [[ "$target" == server ]]; then target=backend; fi
    # Load recorded ports for the target even when it is currently stopped, so
    # `restart backend` keeps 3001 instead of falling back to 3000 and failing
    # the "port changed" guard.
    load_saved_ports
    if [[ "$target" != all ]]; then
        local svc saved_port
        for svc in "${SERVICES[@]}"; do
            if is_running "$svc"; then
                saved_port="$(cat "$PID_DIR/$svc.port" 2>/dev/null || true)"
                if [[ ! "$saved_port" =~ ^[0-9]+$ ]] || (( 10#$saved_port < 1 || 10#$saved_port > 65535 )); then
                    fail "Missing port state for $svc; run restart all once before targeted restarts."
                    return 1
                fi
            fi
        done
    fi
    prepare_start "$target" || return $?
    if [[ "$target" == all ]]; then cmd_stop; else cmd_stop "$target"; fi
    start_services "$target"
}

start_services() {
    local target="${1:-all}"
    if is_running server; then
        fail "The legacy Node gateway is still running; use restart all to switch to the Python entry point."
        return 1
    fi
    if [[ "$target" == all || "$target" == backend ]]; then ensure_database || return 1; fi
    ensure_tls

    # Remember what was requested so we can report any port that got shifted.
    local requested_frontend="$FRONTEND_PORT" requested_backend="$BACKEND_PORT"

    # Local development may share a host: shift an occupied port and record the
    # result in $PID_DIR/<svc>.port. `start` and `restart all` load those files
    # first, so a restart keeps the same port whenever it is still free.
    if [[ "$target" == all || "$target" == backend ]]; then BACKEND_PORT="$(resolve_free_port backend "$BACKEND_PORT")"; fi
    if [[ "$target" == backend && "$BACKEND_PORT" != "$requested_backend" ]]; then fail "Port changed for backend; use restart all to update dependent proxies."; return 1; fi
    if [[ "$target" == all || "$target" == frontend ]]; then FRONTEND_PORT="$(resolve_free_port frontend "$FRONTEND_PORT")"; fi
    if [[ "$target" == frontend && "$FRONTEND_PORT" != "$requested_frontend" ]]; then fail "Port changed for frontend; use restart all to update dependent proxies."; return 1; fi

    cmd_check "$target"

    local scheme="http"
    [[ "$ENABLE_HTTPS" == "1" ]] && scheme="https"
    local authz_tls_insecure="${LENS_MODEL_GATEWAY_AUTHZ_TLS_INSECURE:-}"
    if [[ -z "$authz_tls_insecure" ]]; then
        [[ "$ENABLE_HTTPS" == "1" ]] && authz_tls_insecure=1 || authz_tls_insecure=0
    fi

    if [[ "$target" == all || "$target" == backend ]]; then
    start_one backend env \
        PRISM_MODEL_API_PUBLIC_HOST="$PRISM_MODEL_API_PUBLIC_HOST" \
        PRISM_GPU_PCI_ALLOWLIST="$PRISM_GPU_PCI_ALLOWLIST" \
        PRISM_MCP_URL="$scheme://127.0.0.1:$BACKEND_PORT/api/assistant/mcp" \
        PRISM_MCP_CA_FILE="${TLS_CERT_FILE:-}" \
        LENS_INTERNAL_AUTH_SECRET="$LENS_INTERNAL_AUTH_SECRET" \
        LENS_MODEL_GATEWAY_AUTHZ_TLS_INSECURE="$authz_tls_insecure" \
        BACKEND_PORT="$BACKEND_PORT" BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}" \
        TLS_CERT_FILE="${TLS_CERT_FILE:-}" TLS_KEY_FILE="${TLS_KEY_FILE:-}" TLS_CA_FILE="${TLS_CA_FILE:-}" \
        bash hack/run-backend.sh
    fi

    if [[ "$target" == all || "$target" == frontend ]]; then
    start_one frontend env VITE_HOST="$VITE_HOST" VITE_PORT="$FRONTEND_PORT" \
        VITE_AUTH_PROXY_TARGET="$scheme://127.0.0.1:$BACKEND_PORT" \
        VITE_API_PROXY_TARGET="$scheme://127.0.0.1:$BACKEND_PORT" \
        TLS_CERT_FILE="${TLS_CERT_FILE:-}" TLS_KEY_FILE="${TLS_KEY_FILE:-}" \
        ./node_modules/.bin/vite apps/frontend
    fi

    log "waiting for services to become ready..."
    if [[ "$target" == all || "$target" == backend ]]; then
    wait_for_http backend "$scheme://127.0.0.1:$BACKEND_PORT/api/health" || return 1
    fi
    if [[ "$target" == all || "$target" == frontend ]]; then
    wait_for_http frontend   "$scheme://127.0.0.1:$FRONTEND_PORT/" || return 1
    fi

    log "started service selection: $target; configured endpoints:"
    log "  frontend   $scheme://localhost:$FRONTEND_PORT (bound to $VITE_HOST)"
    log "  app/API    $scheme://127.0.0.1:$BACKEND_PORT"
    if [[ "$ENABLE_HTTPS" == "1" ]]; then
        log "  HTTPS is on with a self-signed cert (browser will warn once; accept it) — needed for mic access etc."
    fi

    if [[ "$FRONTEND_PORT" != "$requested_frontend" || "$BACKEND_PORT" != "$requested_backend" ]]; then
        warn "requested ports were occupied; using free ports instead:"
        warn "  frontend  $requested_frontend -> $FRONTEND_PORT"
        warn "  backend   $requested_backend -> $BACKEND_PORT"
    fi

    print_initial_admin_credentials
}

# ---- stop --------------------------------------------------------------------
cmd_stop() {
    local stopped=0
    local selected=("${SERVICES[@]}" server)
    if [[ $# -gt 0 ]]; then selected=("$@"); fi
    for svc in "${selected[@]}"; do
        local pf
        pf="$(pid_file "$svc")"
        if [[ -f "$pf" ]]; then
            local pid
            pid="$(cat "$pf")"
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                log "stopping $svc (pid $pid)..."
                kill "$pid" 2>/dev/null || true
                stopped=1
            fi
            rm -f "$pf"
        fi
    done
    [[ "$stopped" -eq 1 ]] && sleep 1
    log "selected services stopped: ${selected[*]}"
    return 0
}

# ---- status ------------------------------------------------------------------
cmd_status() {
    for svc in "${SERVICES[@]}"; do
        if is_running "$svc"; then
            log "  $svc: running (pid $(cat "$(pid_file "$svc")"))"
        else
            log "  $svc: stopped"
        fi
    done
}

# ---- clean -------------------------------------------------------------------
# Best-effort graceful stop of a leftover embedded PostgreSQL process for
# <data_dir>. lens-local.sh's backend runs pgserver in-process, so killing the
# uvicorn pid in cmd_stop does not guarantee pgserver's own postgres
# subprocess exits with it -- read its postmaster.pid and signal it directly
# so `clean` doesn't rm -rf a data directory out from under a still-running
# server.
stop_embedded_postgres_process() {
    local data_dir="$1" pid_file pid waited=0
    pid_file="$data_dir/postmaster.pid"
    [[ -f "$pid_file" ]] || return 0
    pid="$(head -n1 "$pid_file" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 0
    log "stopping leftover embedded PostgreSQL (pid $pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null && [[ "$waited" -lt 20 ]]; do
        sleep 0.5
        waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        warn "embedded PostgreSQL (pid $pid) did not stop gracefully — killing it."
        kill -KILL "$pid" 2>/dev/null || true
    fi
}

remove_path() {
    local path="$1"
    if [[ -z "$path" || "$path" == "/" || "$path" == "$HOME" || "$path" == "$ROOT_DIR" ]]; then
        warn "refusing to remove unsafe path: ${path:-<empty>}"
        return 0
    fi
    rm -rf -- "$path"
}

cmd_clean() {
    local clean_option="${1:-}" yes=0 scope="default"
    case "$clean_option" in
        -y|--yes) yes=1; scope="${2:-default}" ;;
        "") ;;
        *) fail "Unknown clean option: $clean_option"; return 2 ;;
    esac
    case "$scope" in default|cache|hard|all) ;; *) fail "Unknown clean scope: $scope"; return 2 ;; esac

    local db_dir bootstrap_file legacy_home
    db_dir="$(python3 -c "import os; print(os.path.expanduser(os.environ.get('LLM_D_BENCH_EMBEDDED_DB_DIR', '~/.llm-d-lens/db/data')))" 2>/dev/null || printf '%s/.llm-d-lens/db/data' "$HOME")"
    bootstrap_file="$(python3 -c "import os; print(os.path.expanduser(os.environ.get('LLM_D_BENCH_DB_BOOTSTRAP_FILE', '~/.llm-d-lens/db-bootstrap.json')))" 2>/dev/null || printf '%s/.llm-d-lens/db-bootstrap.json' "$HOME")"
    legacy_home="$HOME/.llm-d-lens"

    warn "this will stop dev services and permanently delete:"
    warn "  - the embedded database: $db_dir"
    warn "  - local app data:        $LENS_DATA_DIR"
    warn "  - legacy app data:       $legacy_home"
    warn "  - logs/pids/runtime:     $LENS_LOG_DIR, $PID_DIR, $LENS_RUNTIME_DIR"
    if [[ "$scope" == "cache" || "$scope" == "all" ]]; then
        warn "  - local cache:           $LENS_CACHE_DIR (source checkouts, benchmark venvs, temp)"
    fi
    if [[ "$scope" == "hard" || "$scope" == "all" ]]; then
        warn "  - dev environment:       .venv, .dev-tools, .dev-tls, node_modules, build outputs"
    fi
    if [[ "$yes" -ne 1 ]]; then
        local reply
        read -r -p "Type 'yes' to continue: " reply || true
        if [[ "$reply" != "yes" ]]; then
            log "aborted -- nothing was deleted."
            exit 1
        fi
    fi

    cmd_stop
    stop_embedded_postgres_process "$db_dir"

    remove_path "$LENS_DATA_DIR"
    remove_path "$legacy_home"
    remove_path "$db_dir"
    remove_path "$bootstrap_file"
    remove_path "$LENS_LOG_DIR"
    remove_path "$LENS_RUNTIME_DIR"
    remove_path "$PID_DIR"
    remove_path "$ROOT_DIR/.dev-logs"
    remove_path "$ROOT_DIR/.lens-logs"

    if [[ "$scope" == "cache" || "$scope" == "all" ]]; then
        remove_path "$LENS_CACHE_DIR"
        remove_path "$LENS_SCRATCH_DIR"
    fi

    if [[ "$scope" == "hard" || "$scope" == "all" ]]; then
        remove_path "$ROOT_DIR/.venv"
        remove_path "$ROOT_DIR/.dev-tools"
        remove_path "$ROOT_DIR/.dev-tls"
        remove_path "$ROOT_DIR/node_modules"
        remove_path "$ROOT_DIR/apps/frontend/node_modules"
        remove_path "$ROOT_DIR/docs/site/node_modules"
        remove_path "$ROOT_DIR/apps/frontend/dist"
        remove_path "$ROOT_DIR/docs/site/build"
        remove_path "$ROOT_DIR/.pytest_cache"
        remove_path "$ROOT_DIR/apps/backend/.pytest_cache"
    fi

    log "clean complete -- requested local state has been removed."
}

# ---- dispatch ----------------------------------------------------------------
print_usage() {
    cat <<'EOF'
Usage: hack/lens-local.sh COMMAND [OPTIONS]

Local source installation and service management:
  check          Validate the environment without starting services.
  start          Prepare dependencies and start the local services.
  restart [all|frontend|server|backend]
                 Complete preflight checks, then restart the selected services.
  stop           Stop services started by this launcher.
  status         Show running service PIDs.
  install-node   Install a supported Node.js into .dev-tools/.
  clean [-y|--yes] [--cache|--hard|--all]
                 Stop services and permanently delete the embedded database,
                 local app data, and launcher logs/PIDs. Requires confirmation
                 unless -y/--yes is supplied.
                   --cache  also delete the local cache (source checkouts,
                            benchmark venvs, temporary files)
                   --hard   also delete the project .venv, .dev-tools, .dev-tls,
                            node_modules and build outputs
                   --all    delete app state, caches and the dev environment

Options for check, start and restart:
  --target-node NODE   Override TARGET_NODE; exported as PRISM_K8S_TARGET_NODE.
  --kubeconfig FILE    Override KUBECONFIG. Relative to the invocation directory.
  -h, --help          Show this help.

No target-node configuration file is loaded. Without a target node, no
Kubernetes preflight is performed. Without KUBECONFIG, Kubernetes uses its
normal configuration lookup. An inherited KUBECONFIG file list is preserved.
TARGET_NODE falls back to the existing PRISM_K8S_TARGET_NODE environment value.
stop/status never require a Kubernetes connection.

Environment:
    FRONTEND_PORT / BACKEND_PORT   Defaults: 5173 / 3000 (SERVER_PORT is a legacy alias)
  LLM_D_ROOT              llm-d checkout with guides/; default $LENS_CACHE_DIR/llm-d
  PRISM_GPU_PCI_ALLOWLIST  Optional explicit device restriction; no auto-discovery
  ENABLE_HTTPS            Default 1; set 0 for plain HTTP
  TLS_HOST / TLS_DIR      Certificate host (auto-detected) / directory (.dev-tls)
  LENS_MODEL_GATEWAY_AUTHZ_TLS_INSECURE
                          Default 1 with HTTPS, 0 otherwise; skips authz TLS
                          certificate verification (trusted local/dev only)
  NODE_MIN_MAJOR          Default 22
  NODE_AUTO_INSTALL       Default 1; set 0 to disable automatic Node downloads
  NODE_INSTALL_VERSION   Optional exact Node.js version
EOF
}

lens_local_main() {
    local action="${1:-}" clean_scope="default" clean_yes="" option cluster_options=0 service_target=all service_selected=0
    if [[ -z "$action" ]]; then
        print_usage
        return 1
    fi
    shift
    case "$action" in
        -h|--help|help) print_usage; return 0 ;;
        check|start|restart|stop|status|clean|install-node) ;;
        *) fail "Unknown command: $action"; print_usage >&2; return 2 ;;
    esac

    TARGET_NODE="${TARGET_NODE:-${PRISM_K8S_TARGET_NODE:-}}"
    while [[ $# -gt 0 ]]; do
        option="$1"
        case "$option" in
            all|frontend|server|backend)
                [[ "$action" == restart && "$service_selected" -eq 0 ]] || { fail "A service target is only valid once for restart."; return 2; }
                service_target="$option"
                service_selected=1
                shift
                ;;
            --target-node|--kubeconfig)
                if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
                    fail "$option requires a nonempty value."
                    return 2
                fi
                cluster_options=1
                if [[ "$option" == "--target-node" ]]; then
                    TARGET_NODE="$2"
                else
                    KUBECONFIG="$2"
                    [[ "$KUBECONFIG" == /* ]] || KUBECONFIG="$LAUNCH_DIR/$KUBECONFIG"
                    export KUBECONFIG
                fi
                shift 2
                ;;
            -y|--yes)
                [[ "$action" == clean ]] || { fail "$option is only valid for clean."; return 2; }
                clean_yes="$option"
                shift
                ;;
            --cache|--hard|--all)
                [[ "$action" == clean ]] || { fail "$option is only valid for clean."; return 2; }
                local requested_scope="${option#--}"
                if [[ "$clean_scope" != "default" && "$clean_scope" != "$requested_scope" ]]; then
                    fail "Choose only one of --cache, --hard or --all."
                    return 2
                fi
                clean_scope="$requested_scope"
                shift
                ;;
            -h|--help) print_usage; return 0 ;;
            *) fail "Unknown option: $option"; return 2 ;;
        esac
    done
    if [[ "$cluster_options" -eq 1 && "$action" != check && "$action" != start && "$action" != restart ]]; then
        fail "Target-node and kubeconfig options are only valid for check, start and restart."
        return 2
    fi
    case "$action" in
        check|start|restart)
            export TARGET_NODE PRISM_K8S_TARGET_NODE="$TARGET_NODE"
            [[ -z "${KUBECONFIG+x}" ]] || export KUBECONFIG
            ;;
    esac
    case "$action" in
        check) prepare_target_node && cmd_check ;;
        start) cmd_start ;;
        restart) cmd_restart "$service_target" ;;
        stop) cmd_stop ;;
        status) cmd_status ;;
        clean) cmd_clean "$clean_yes" "$clean_scope" ;;
        install-node) ensure_node ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    lens_local_main "$@"
fi
