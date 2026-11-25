#!/usr/bin/env bash
# Download a pinned Lens source tree and install it on a supported host.
# Parse the complete body before execution, including when streamed into Bash.
{
set -euo pipefail

# Maintained by the "release stamp" command before tagging a release.
LENS_SOURCE_VERSION="main"
REPOSITORY="llm-d-incubation/llm-d-lens"
PIPE_ENTRY=0
[[ -n "${BASH_SOURCE[0]:-}" ]] || PIPE_ENTRY=1

# ---- Host OS header ----------------------------------------------------------
# The only per-OS part of this installer; every other step calls hostos.
INSTALLER_OS_ID="ubuntu"
INSTALLER_FILE_NAME="LensInstaller-Ubuntu-x86_64-online.sh"
INSTALLER_DISPLAY_NAME="Ubuntu 22.04+ (x86_64)"
INSTALLER_PREREQ_HINT="sudo apt-get install curl ca-certificates python3"

online_fail() { printf '[lens-online] %s\n' "$*" >&2; exit 1; }

online_help() {
    cat <<EOF
Lens online installer for $INSTALLER_DISPLAY_NAME
Source: $LENS_SOURCE_VERSION

Usage: bash $INSTALLER_FILE_NAME [options]
  --silent, -y          Install without prompts.
  --install-dir DIR     New install directory (default: \$HOME/llm-d-lens).
  --app-port PORT       Web app/API port (default: 3000).
    --backend-port PORT   Legacy alias for --app-port; the Python app serves both UI and API.
  --no-https            Use HTTP instead of self-signed HTTPS.
  --tls-host HOST       Host/IP for the HTTPS certificate.
  --no-start            Install without starting services.
  --no-apt              Do not install system packages with apt.
  --db-mode MODE        embedded (default) or external.
  --db-engine ENGINE    postgresql, mysql, oracle, or mssql.
  --db-host HOST        External database host.
  --db-port PORT        External database port.
  --db-name NAME        External database name.
  --db-user USER        External database username.
  --db-password VALUE   External database password (prefer DB_PASSWORD env).
  --db-embedded-password VALUE
                        Embedded database password (prefer
                        DB_EMBEDDED_PASSWORD env).
  -h, --help            Show help without downloading anything.

The downloaded script selects the version; there is no version override.
Fresh installs only. Use the installed lensctl.sh for management/uninstall.
Run as the intended service user; sudo is used only when installing missing system packages.
Piped installs read prompts from /dev/tty and, after source validation, save a
uniquely named installer copy in the current directory.
For automation, pipe to: bash -s -- --silent [options]

Maintainers only (edit/check this script, without installing):
  release stamp VERSION   Embed main or a vMAJOR.MINOR.PATCH release tag.
  release check VERSION   Verify the embedded version matches the branch/tag.
EOF
}

# Stage 1 has only the extracted source archive, not an installed virtualenv.
ONLINE_ARCHIVE_ROOT=""
online_hostos_exec() {
    PYTHONPATH="$ONLINE_ARCHIVE_ROOT/apps/backend/src" \
        python3 -m llm_d_lens.infrastructure.hostos "$@"
}

# online_hostos <args...> — run hostos and return 0/1. Exit >= 2 means a
# usage/internal error: print it and abort instead of misreading it.
online_hostos() {
    local rc=0
    online_hostos_exec "$@" || rc=$?
    if (( rc >= 2 )); then
        fail "hostos command failed (exit $rc)"
        exit 1
    fi
    return "$rc"
}

online_main() {
    local install_dir="${INSTALL_DIR:-$HOME/llm-d-lens}" silent=0 explicit_dir=0
    local args=() option value
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) online_help; return ;;
            --silent|-y) silent=1; args+=("$1"); shift ;;
            --no-start|--no-apt|--no-https) args+=("$1"); shift ;;
            --install-dir|--install-dir=*|--app-port|--app-port=*|--backend-port|--backend-port=*|\
            --tls-host|--tls-host=*|--db-mode|--db-mode=*|--db-engine|--db-engine=*|\
            --db-host|--db-host=*|--db-port|--db-port=*|--db-name|--db-name=*|\
            --db-user|--db-user=*|--db-password|--db-password=*|\
            --db-embedded-password|--db-embedded-password=*)
                option="${1%%=*}"
                if [[ "$1" == *=* ]]; then
                    value="${1#*=}"; shift
                else
                    [[ $# -ge 2 && "$2" != --* ]] || online_fail "Missing value for $option."
                    value="$2"; shift 2
                fi
                [[ -n "$value" ]] || online_fail "Empty value for $option."
                if [[ "$option" == --install-dir ]]; then
                    install_dir="$value"; explicit_dir=1
                else
                    args+=("$option" "$value")
                fi ;;
            *) online_fail "Unsupported option: $1. See --help; upgrades and version overrides are not supported." ;;
        esac
    done
    [[ "$LENS_SOURCE_VERSION" == main || "$LENS_SOURCE_VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]] ||
        online_fail "Invalid embedded source version."
    command -v python3 >/dev/null ||
        online_fail "Missing python3. Install prerequisites with: $INSTALLER_PREREQ_HINT"
    command -v curl >/dev/null || command -v wget >/dev/null ||
        online_fail "Install curl or wget before running the online installer."
    if [[ "$silent" == 0 ]]; then
        if [[ ! -t 0 && "$PIPE_ENTRY" == 1 ]]; then
            if { exec 3</dev/tty; } 2>/dev/null; then
                exec 0<&3 3<&-
            else
                online_fail "No controlling terminal. For automation, pipe to bash -s -- --silent."
            fi
        fi
        [[ -t 0 ]] || online_fail "Interactive installation requires a terminal; use --silent."
        if [[ "$explicit_dir" == 0 ]]; then
            read -r -p "Install directory [$install_dir]: " value
            install_dir="${value:-$install_dir}"
        fi
    fi
    install_dir="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$install_dir")"
    [[ ! -L "$install_dir" ]] || online_fail "The install directory must not be a symbolic link."
    if [[ -e "$install_dir" ]]; then
        [[ -d "$install_dir" && -z "$(find "$install_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] ||
            online_fail "Install directory is not empty: $install_dir. Online upgrades are not supported."
    fi

    umask 077
    local log_dir="${LENS_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/lens/logs}"
    local scratch="${LENS_SCRATCH_DIR:-${LENS_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/lens}/tmp}"
    mkdir -p "$log_dir" "$scratch"
    LOG_FILE="$(mktemp "$log_dir/online-install-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")"
    WORK_DIR="$(mktemp -d "$scratch/online-install-XXXXXXXX")"
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    printf '[lens-online] Source: %s; log: %s\n' "$LENS_SOURCE_VERSION" "$LOG_FILE"
    local source_ref="heads/main" sha archive_root
    [[ "$LENS_SOURCE_VERSION" == main ]] || source_ref="tags/$LENS_SOURCE_VERSION"
    download "https://api.github.com/repos/$REPOSITORY/commits/refs/$source_ref" "$WORK_DIR/commit.json"
    sha="$(python3 - "$WORK_DIR/commit.json" <<'PY'
import json
import re
import sys

with open(sys.argv[1]) as source:
    sha = json.load(source).get("sha", "")
if not re.fullmatch(r"[0-9a-f]{40}", sha):
    sys.exit("GitHub did not return a valid commit SHA.")
print(sha)
PY
    )"
    printf '[lens-online] Resolved %s to %s\n' "$LENS_SOURCE_VERSION" "$sha" | tee -a "$LOG_FILE"
    download "https://codeload.github.com/$REPOSITORY/tar.gz/$sha" "$WORK_DIR/source.tar.gz"
    archive_root="$WORK_DIR/llm-d-lens-$sha"
    python3 - "$WORK_DIR/source.tar.gz" "$WORK_DIR" "llm-d-lens-$sha" "$INSTALLER_FILE_NAME" <<'PY'
import pathlib
import sys
import tarfile

archive, destination, root, installer = sys.argv[1:]
with tarfile.open(archive, "r:gz") as source:
    members = source.getmembers()
    seen = set()
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or path.parts[0] != root or str(path) in seen
                or not (member.isdir() or member.isfile())):
            sys.exit(f"Unsafe or unsupported source archive entry: {member.name}")
        seen.add(str(path))
    required = ("package.json", "apps/backend/pyproject.toml", "hack/storage-env.sh",
                f"scripts/{installer}",
                "apps/backend/src/llm_d_lens/infrastructure/hostos/__main__.py")
    for name in required:
        if not any(m.name == f"{root}/{name}" and m.isfile() for m in members):
            sys.exit(f"Source archive is missing {name}.")
    for member in members:
        member.mode &= 0o777
        source.extract(member, destination, set_attrs=False)
        target = pathlib.Path(destination, member.name)
        target.chmod(member.mode)
PY
    ONLINE_ARCHIVE_ROOT="$archive_root"
    local host_check
    if ! host_check="$(online_hostos check --installer "$INSTALLER_OS_ID" 2>&1)"; then
        online_fail "$host_check"
    fi
    if [[ "$PIPE_ENTRY" == 1 ]]; then
        local saved_script
        saved_script="$(mktemp "$PWD/${INSTALLER_FILE_NAME%.sh}.XXXXXXXX.sh")"
        if ! cp -- "$archive_root/scripts/$INSTALLER_FILE_NAME" "$saved_script"; then
            rm -f -- "$saved_script"
            online_fail "Could not save the installer in $PWD."
        fi
        printf '[lens-online] Installer saved: %s\n' "$saved_script" | tee -a "$LOG_FILE"
    fi
    # This internal mode fixes the chosen path and rechecks it before any writes.
    if bash "$archive_root/scripts/$INSTALLER_FILE_NAME" \
        --install-source --fresh-install --install-dir "$install_dir" "${args[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        python3 - "$install_dir" "$LENS_SOURCE_VERSION" "$sha" <<'PY'
import datetime
import json
import pathlib
import sys

directory, version, sha = sys.argv[1:]
record = {"sourceVersion": version, "commit": sha,
          "installedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()}
pathlib.Path(directory, ".lens-version.json").write_text(json.dumps(record, indent=2) + "\n")
PY
        printf '[lens-online] Installed %s (%s). Log: %s\n' "$LENS_SOURCE_VERSION" "$sha" "$LOG_FILE"
    else
        return 1
    fi
}

download() {
    if command -v curl >/dev/null; then
        curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
            --retry 3 --connect-timeout 15 --max-time 600 \
            --output "$2" "$1" 2>>"$LOG_FILE" ||
            online_fail "Download failed: $1. Check network/proxy or GitHub API rate limits. Log: $LOG_FILE"
    else
        timeout 600 wget --tries=3 --timeout=30 --max-redirect=0 \
            --output-document="$2" "$1" 2>>"$LOG_FILE" ||
            online_fail "Download failed: $1. Check network/proxy or GitHub API rate limits. Log: $LOG_FILE"
    fi
}

cleanup() {
    local result=$?
    [[ "$result" == 0 ]] || printf '[lens-online] Installation failed. Log: %s\n' "$LOG_FILE" >&2
    rm -rf -- "$WORK_DIR"
    exit "$result"
}

# ---- Installation from the downloaded source tree -----------------------------
# ---- Paths -------------------------------------------------------------------
SCRIPT_DIR=""
SOURCE_DIR=""
if [[ "$PIPE_ENTRY" == 0 ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# ---- Colors & styling (Copilot-CLI-esque) -------------------------------------
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
    RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    MAGENTA=$'\033[35m'; CYAN=$'\033[36m'; WHITE=$'\033[37m'
    # "llm-" is a muted grey, "d" is a deep plum/purple — the brand mark.
    LOGO_GRAY=$'\033[38;5;245m'; LOGO_PLUM=$'\033[38;5;97m'
else
    BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; MAGENTA=""; CYAN=""; WHITE=""
    LOGO_GRAY=""; LOGO_PLUM=""
fi

info()  { printf '%s[lens]%s %s\n' "$CYAN$BOLD" "$RESET" "$*"; }
ok()    { printf '%s[lens]%s %s✔%s %s\n' "$CYAN$BOLD" "$RESET" "$GREEN" "$RESET" "$*"; }
warn()  { printf '%s[lens]%s %s⚠%s %s\n' "$YELLOW$BOLD" "$RESET" "$YELLOW" "$RESET" "$*" >&2; }
fail()  { printf '%s[lens]%s %s✖%s %s\n' "$RED$BOLD" "$RESET" "$RED" "$RESET" "$*" >&2; }
step()  { printf '\n%s➜ %s%s\n' "$MAGENTA$BOLD" "$*" "$RESET"; }

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
            printf '%s\n' "$lines" | sed "s/^/  ${DIM}/;s/\$/${RESET}/"
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

print_logo() {
    [[ "${QUIET_LOGO:-0}" == "1" ]] && return 0
    local logo=(
        ' _ _                     _       _                '
        '| | |_ __ ___         __| |     | | ___ _ __  ___ '
        "| | | '_ \` _ \\ _____ / _\` |_____| |/ _ \\ '_ \\/ __|"
        '| | | | | | | |_____| (_| |_____| |  __/ | | \__ \'
        '|_|_|_| |_| |_|      \__,_|     |_|\___|_| |_|___/'
    )
    echo
    # Column 21 is where "llm-" ends and "-lens" begins in the rendered glyphs.
    for line in "${logo[@]}"; do
        printf '  %s%s%s%s%s%s\n' \
            "$LOGO_GRAY$BOLD" "${line:0:21}" "$RESET" \
            "$LOGO_PLUM$BOLD" "${line:21}" "$RESET"
    done
    # A plain, understated bar beneath the wordmark — matching its two tones.
    printf '  %s▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇%s%s▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇%s\n' "$LOGO_GRAY" "$RESET" "$LOGO_PLUM" "$RESET"
    printf '  %sLLM-D-LENS%s %sInstaller for %s%s\n\n' "$BOLD$WHITE" "$RESET" "$DIM" "$INSTALLER_DISPLAY_NAME" "$RESET"
}

# ---- Defaults / configuration -------------------------------------------------
SILENT=0
INSTALL_DIR_DEFAULT="$HOME/llm-d-lens"
INSTALL_DIR="${INSTALL_DIR:-$INSTALL_DIR_DEFAULT}"
APP_PORT="${APP_PORT:-${BACKEND_PORT:-${SERVER_PORT:-3000}}}"
BACKEND_PORT="$APP_PORT"
START_AFTER_INSTALL=1
ALLOW_SYSTEM_PACKAGES=1
FRESH_INSTALL=0
UNINSTALL=0
UNINSTALL_ARGS=()
# HTTPS on by default: browsers only expose mic access etc. in a "secure
# context" (HTTPS or http://localhost), so a bare-metal box reached over
# http://<lan-ip> needs it. A self-signed cert is generated automatically.
ENABLE_HTTPS=1
TLS_HOST="${TLS_HOST:-}"

# Application-level ("Step 0") database configuration -- see
# llm_d_lens/db/bootstrap_cli.py. Silent installs always default to the
# built-in embedded PostgreSQL unless --db-mode/--db-* flags override it;
# interactive installs are asked via db_interactive_configure() below.
DB_MODE="${DB_MODE:-embedded}"
DB_ENGINE="${DB_ENGINE:-postgresql}"
DB_HOST="${DB_HOST:-}"
DB_PORT="${DB_PORT:-}"
DB_NAME="${DB_NAME:-}"
DB_USER="${DB_USER:-}"
DB_PASSWORD="${DB_PASSWORD:-}"
# Password enforced on the built-in embedded PostgreSQL's superuser (only
# used when DB_MODE=embedded; see llm_d_lens/db/settings.py's
# _harden_embedded_postgres_password). Unlike hack/lens-local.sh's local-dev
# embedded database (intentionally password-less), the installer always
# requires one -- prompted interactively, or auto-generated for silent
# installs if not supplied via --db-embedded-password.
DB_EMBEDDED_PASSWORD="${DB_EMBEDDED_PASSWORD:-}"
# Set to 1 by configure_database() when DB_EMBEDDED_PASSWORD above was
# auto-generated (not supplied by the user) -- the final "Done" summary uses
# this to decide whether the password needs to be shown/reminded about, so
# it isn't only buried in the mid-install log output.
DB_EMBEDDED_PASSWORD_AUTOGEN=0

NODE_MIN_MAJOR="${NODE_MIN_MAJOR:-22}"
NODE_FALLBACK_VERSION="${NODE_FALLBACK_VERSION:-v22.23.2}"
NODE_DIST_URL="${NODE_DIST_URL:-https://nodejs.org/dist}"
PYTHON_BIN="${PYTHON_BIN:-}"
UV_VERSION="0.13.0"

print_help() {
    online_help
}

# ---- Argument parsing ----------------------------------------------------------
parse_args() {
while [[ $# -gt 0 ]]; do
    case "$1" in
        --silent|-y)       SILENT=1; shift ;;
        --install-dir)     INSTALL_DIR="$2"; shift 2 ;;
        --install-dir=*)   INSTALL_DIR="${1#*=}"; shift ;;
        --app-port)        require_valid_port "$2" "--app-port"; APP_PORT="$2"; shift 2 ;;
        --app-port=*)      require_valid_port "${1#*=}" "--app-port"; APP_PORT="${1#*=}"; shift ;;
        --backend-port)    require_valid_port "$2" "--backend-port"; APP_PORT="$2"; BACKEND_PORT="$APP_PORT"; shift 2 ;;
        --backend-port=*)  require_valid_port "${1#*=}" "--backend-port"; APP_PORT="${1#*=}"; BACKEND_PORT="$APP_PORT"; shift ;;
        --no-https)        ENABLE_HTTPS=0; shift ;;
        --tls-host)        TLS_HOST="$2"; shift 2 ;;
        --tls-host=*)      TLS_HOST="${1#*=}"; shift ;;
        --no-start)         START_AFTER_INSTALL=0; shift ;;
        --no-apt)           ALLOW_SYSTEM_PACKAGES=0; shift ;;
        --fresh-install)    FRESH_INSTALL=1; shift ;;
        --db-mode)          DB_MODE="$2"; shift 2 ;;
        --db-mode=*)        DB_MODE="${1#*=}"; shift ;;
        --db-engine)        DB_ENGINE="$2"; shift 2 ;;
        --db-engine=*)      DB_ENGINE="${1#*=}"; shift ;;
        --db-host)          DB_HOST="$2"; shift 2 ;;
        --db-host=*)        DB_HOST="${1#*=}"; shift ;;
        --db-port)          DB_PORT="$2"; shift 2 ;;
        --db-port=*)        DB_PORT="${1#*=}"; shift ;;
        --db-name)          DB_NAME="$2"; shift 2 ;;
        --db-name=*)        DB_NAME="${1#*=}"; shift ;;
        --db-user)          DB_USER="$2"; shift 2 ;;
        --db-user=*)        DB_USER="${1#*=}"; shift ;;
        --db-password)      DB_PASSWORD="$2"; shift 2 ;;
        --db-password=*)    DB_PASSWORD="${1#*=}"; shift ;;
        --db-embedded-password)    DB_EMBEDDED_PASSWORD="$2"; shift 2 ;;
        --db-embedded-password=*)  DB_EMBEDDED_PASSWORD="${1#*=}"; shift ;;
        --uninstall)        UNINSTALL=1; shift ;;
        --keep-data)        UNINSTALL_ARGS+=("--keep-data"); shift ;;
        --purge)            UNINSTALL_ARGS+=("--purge"); shift ;;
        --keep-db)          UNINSTALL_ARGS+=("--keep-db"); shift ;;
        --purge-db)         UNINSTALL_ARGS+=("--purge-db"); shift ;;
        -h|--help)          print_help; exit 0 ;;
        *) fail "unknown option: $1"; print_help; exit 1 ;;
    esac
done
}

# ---- Environment gates ---------------------------------------------------------
# hostos_exec runs the host layer from the downloaded source tree with the
# system python3. It is the low-level command tests can stub.
hostos_exec() {
    PYTHONPATH="$SOURCE_DIR/apps/backend/src" \
        python3 -m llm_d_lens.infrastructure.hostos "$@"
}

# hostos <args...> — run hostos and return 0/1. Exit >= 2 means a usage or
# internal error: print it and abort instead of misreading the result.
hostos() {
    local rc=0
    hostos_exec "$@" || rc=$?
    if (( rc >= 2 )); then
        fail "hostos command failed (exit $rc)"
        exit 1
    fi
    return "$rc"
}

require_host_os() {
    local message
    if ! message="$(hostos check --installer "$INSTALLER_OS_ID")"; then
        exit 1
    fi
    ok "$message"
}

# ---- Port conflict handling -------------------------------------------------------
# Ports are passed to hostos as plain numbers, so validate user input first: a
# typo must not reach argparse (which would abort the install with a usage error).
is_valid_port() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 >= 1 && 10#$1 <= 65535 ))
}

require_valid_port() {
    local value="$1" label="$2"
    if ! is_valid_port "$value"; then
        fail "$label must be a port between 1 and 65535 (got '$value')."
        exit 1
    fi
}

# This host may be shared with other users/instances, so a chosen port
# (default or explicitly requested) can already be bound by someone else.
# Relies on an earlier `hostos check` gate: `port-in-use` exits 1 both when the port is free and when unsupported.
port_in_use() {
    hostos port-in-use -- "$1"
}

# find_free_port <start> -> echoes the first free port at or after <start>.
find_free_port() {
    hostos free-port -- "$1"
}

# describe_port_owner <port> -> best-effort "who's holding this port" string.
# On this kind of shared host, the occupant is often another user's process,
# whose PID/command won't be visible without root — say so plainly instead
# of pretending we know, so the user isn't left guessing why a pick failed.
describe_port_owner() {
    hostos port-owner -- "$1"
}

# ask_port <prompt> <default> -> like ask(), but loops until the chosen port
# is valid and actually free, suggesting the next free one as the new default
# each time and explaining what's occupying the rejected port.
ask_port() {
    local prompt="$1" default="$2" port suggestion owner
    while true; do
        port="$(ask "$prompt" "$default")"
        if ! is_valid_port "$port"; then
            warn "port '$port' is not valid; enter a port between 1 and 65535."
            continue
        fi
        if port_in_use "$port"; then
            owner="$(describe_port_owner "$port")" || exit 1
            suggestion="$(find_free_port $((port + 1)))" || exit 1
            warn "port $port is already held by $owner — try $suggestion, or another port of your own choice."
            default="$suggestion"
            continue
        fi
        printf '%s\n' "$port"
        return 0
    done
}

# resolve_ports_silently — non-interactive equivalent of ask_port: if a
# chosen port (default or --app-port/--backend-port) is already taken,
# switch to the next free one instead of failing, but always explain why
# (which process is occupying it, where discoverable) and surface which
# ports actually ended up being used (see main()'s "silent mode" info line).
resolve_ports_silently() {
    require_valid_port "$APP_PORT" "app port"
    local original owner
    if port_in_use "$APP_PORT"; then
        original="$APP_PORT"
        owner="$(describe_port_owner "$original")" || exit 1
        warn "app port $original is already held by $owner."
        APP_PORT="$(find_free_port "$APP_PORT")" || exit 1
        warn "switched app port to $APP_PORT instead."
    fi
    BACKEND_PORT="$APP_PORT"
}

# ---- Interactive prompts ---------------------------------------------------------
ask() {
    # ask <prompt> <default> -> echoes the chosen value
    local prompt="$1" default="$2" reply
    read -r -p "$(printf '%s?%s %s %s[%s]%s ' "$CYAN$BOLD" "$RESET" "$prompt" "$DIM" "$default" "$RESET")" reply || true
    printf '%s\n' "${reply:-$default}"
}

confirm() {
    local prompt="$1" default="${2:-Y}" reply
    read -r -p "$(printf '%s?%s %s %s[%s/%s]%s ' "$CYAN$BOLD" "$RESET" "$prompt" "$DIM" \
        "$([[ $default == Y ]] && echo Y || echo y)" "$([[ $default == N ]] && echo N || echo n)" "$RESET")" reply || true
    reply="${reply:-$default}"
    [[ "$reply" =~ ^[Yy] ]]
}

db_interactive_configure() {
    echo
    info "Database (used for Lens's own structured data -- clusters, deployments, etc.)"
    if confirm "Use built-in embedded PostgreSQL" "Y"; then
        DB_MODE="embedded"
        echo
        info "The embedded database is only reachable via a local Unix socket, but a password is still required (defense in depth)."
        read -r -s -p "$(printf '%s?%s Embedded database password %s(leave blank to auto-generate)%s ' "$CYAN$BOLD" "$RESET" "$DIM" "$RESET")" DB_EMBEDDED_PASSWORD || true
        echo
        return 0
    fi
    DB_MODE="external"
    echo "    engines: postgresql, mysql, oracle, mssql"
    while true; do
        DB_ENGINE="$(ask "Database engine" "$DB_ENGINE")"
        local default_port
        case "$DB_ENGINE" in
            mysql)      default_port="3306" ;;
            oracle)     default_port="1521" ;;
            mssql)      default_port="1433" ;;
            *)          default_port="5432" ;;
        esac
        DB_HOST="$(ask "Database host" "${DB_HOST}")"
        DB_PORT="$(ask "Database port" "${DB_PORT:-$default_port}")"
        DB_NAME="$(ask "Database name" "${DB_NAME:-lens}")"
        DB_USER="$(ask "Database username" "${DB_USER}")"
        read -r -s -p "$(printf '%s?%s Database password %s(hidden)%s ' "$CYAN$BOLD" "$RESET" "$DIM" "$RESET")" DB_PASSWORD || true
        echo

        if test_external_db_connection; then
            ok "connected to $DB_ENGINE at $DB_HOST:$DB_PORT/$DB_NAME"
            return 0
        fi

        if confirm "Use the built-in embedded PostgreSQL instead" "N"; then
            DB_MODE="embedded"
            echo
            info "The embedded database is only reachable via a local Unix socket, but a password is still required (defense in depth)."
            read -r -s -p "$(printf '%s?%s Embedded database password %s(leave blank to auto-generate)%s ' "$CYAN$BOLD" "$RESET" "$DIM" "$RESET")" DB_EMBEDDED_PASSWORD || true
            echo
            return 0
        fi
        info "let's try the external database details again."
    done
}

# test_external_db_connection — probes the DB_* connection details the user
# just entered before letting the installer proceed. Needs the app's own
# virtualenv (SQLAlchemy + drivers), so it provisions the venv early (both
# sync_source and ensure_venv are idempotent and safe to call again later in
# main()'s normal flow). Returns 0 iff the connection succeeds; on failure it
# prints the error and returns 1 so the caller can offer a retry.
test_external_db_connection() {
    ensure_system_packages
    TOOLS_DIR="$INSTALL_DIR/.tools"
    ensure_python
    sync_source
    ensure_venv
    info "testing connection to $DB_ENGINE at $DB_HOST:$DB_PORT/$DB_NAME ..."
    local args=(--test-connection --mode external --engine "$DB_ENGINE" --host "$DB_HOST" --dbname "$DB_NAME")
    [[ -n "$DB_PORT" ]] && args+=(--port "$DB_PORT")
    [[ -n "$DB_USER" ]] && args+=(--username "$DB_USER")
    [[ -n "$DB_PASSWORD" ]] && args+=(--password "$DB_PASSWORD")
    if ! (cd "$INSTALL_DIR" && .venv/bin/python -m llm_d_lens.db.bootstrap_cli "${args[@]}"); then
        fail "could not connect to that database -- double check the host/port/credentials and that it's reachable from this machine."
        return 1
    fi
}

interactive_configure() {
    step "Configuration"
    info "Press Enter to accept the default shown in brackets."
    if [[ "$FRESH_INSTALL" != 1 ]]; then
        INSTALL_DIR="$(ask "Install directory" "$INSTALL_DIR")"
    fi
    APP_PORT="$(ask_port "Web app / API port" "$APP_PORT")"
    BACKEND_PORT="$APP_PORT"
    if confirm "Enable HTTPS (self-signed cert; needed for browser mic access etc. off localhost)" "Y"; then
        ENABLE_HTTPS=1
    else
        ENABLE_HTTPS=0
    fi

    db_interactive_configure

    echo
    info "Summary:"
    printf '    install dir     %s%s%s\n' "$BOLD" "$INSTALL_DIR" "$RESET"
    printf '    app port        %s%s%s\n' "$BOLD" "$APP_PORT" "$RESET"
    printf '    https           %s%s%s\n' "$BOLD" "$([[ "$ENABLE_HTTPS" == "1" ]] && echo enabled || echo disabled)" "$RESET"
    printf '    database        %s%s%s\n' "$BOLD" "$([[ "$DB_MODE" == "embedded" ]] && echo "built-in embedded PostgreSQL" || echo "$DB_ENGINE at $DB_HOST:$DB_PORT/$DB_NAME")" "$RESET"
    echo
    if ! confirm "Proceed with installation" "Y"; then
        warn "aborted by user."
        exit 1
    fi
    if confirm "Start Lens after installing" "Y"; then
        START_AFTER_INSTALL=1
    else
        START_AFTER_INSTALL=0
    fi
}

# ---- System package prerequisites ------------------------------------------------
ensure_system_packages() {
    step "Checking system packages"
    local output missing=() package
    output="$(hostos packages missing)" || exit 1
    while IFS= read -r package; do
        [[ -n "$package" ]] && missing+=("$package")
    done <<< "$output"

    if [[ ${#missing[@]} -eq 0 ]]; then
        ok "all required system packages are present"
        return 0
    fi

    local list="${missing[*]}"
    warn "missing system packages: $list"

    if [[ "$ALLOW_SYSTEM_PACKAGES" != "1" ]]; then
        local hint
        hint="$(hostos packages hint "${missing[@]}")" || exit 1
        fail "cannot install missing packages (--no-apt set). Install manually: $hint"
        exit 1
    fi

    local commands
    commands="$(hostos packages show "${missing[@]}")" || exit 1
    if [[ "$SILENT" != "1" ]]; then
        confirm "Install missing packages with '$commands' now" "Y" ||
            { fail "cannot continue without required packages."; exit 1; }
    fi

    info "running: $commands"
    if ! hostos packages install "${missing[@]}"; then
        fail "system package installation failed."
        exit 1
    fi
    ok "system packages installed"
}

# ---- Node.js toolchain (adapted from hack/lens-local.sh) ------------------------------
TOOLS_DIR="$INSTALL_DIR/.tools"
LOCAL_NODE_DIR="$TOOLS_DIR/node"

node_major() {
    command -v node >/dev/null 2>&1 || return 0
    node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1
}

node_is_supported() {
    local major
    major="$(node_major)"
    [[ -n "$major" ]] && [[ "$major" -ge "$NODE_MIN_MAJOR" ]]
}

activate_local_node() {
    [[ -x "$LOCAL_NODE_DIR/bin/node" ]] || return 1
    case ":$PATH:" in
        *":$LOCAL_NODE_DIR/bin:"*) ;;
        *) PATH="$LOCAL_NODE_DIR/bin:$PATH"; export PATH ;;
    esac
    return 0
}

resolve_node_version() {
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

# sha256_file <path> — print the SHA-256 hex digest using Python's hashlib.
sha256_file() {
    python3 - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

install_node() {
    local platform version tarball url workdir
    for tool in curl tar; do
        command -v "$tool" >/dev/null 2>&1 || { fail "$tool is required to install Node.js"; return 1; }
    done
    command -v python3 >/dev/null 2>&1 ||
        { fail "python3 is required to verify the Node.js download"; return 1; }

    platform="$(hostos tool-platform --style node)" || return 1
    version="$(resolve_node_version)"
    tarball="node-$version-$platform.tar.xz"
    url="$NODE_DIST_URL/$version/$tarball"
    info "installing Node.js $version into $LOCAL_NODE_DIR ..."

    mkdir -p "$TOOLS_DIR"
    workdir="$(mktemp -d "$TOOLS_DIR/node-download.XXXXXX")"
    # shellcheck disable=SC2064
    trap "rm -rf '$workdir'" RETURN

    curl --proto '=https' --tlsv1.2 -fL --progress-bar -m 600 -o "$workdir/$tarball" "$url" \
        || { fail "failed to download $url"; return 1; }
    curl --proto '=https' --tlsv1.2 -fL -sS -m 60 -o "$workdir/SHASUMS256.txt" "$NODE_DIST_URL/$version/SHASUMS256.txt" \
        || { fail "failed to download the checksum list for $version"; return 1; }

    local expected actual
    expected="$(awk -v file="$tarball" '$2 == file {print $1}' "$workdir/SHASUMS256.txt")"
    [[ -n "$expected" ]] || { fail "$tarball is not listed in SHASUMS256.txt"; return 1; }
    actual="$(sha256_file "$workdir/$tarball" 2>/dev/null || true)"
    [[ "$expected" == "$actual" ]] || { fail "checksum mismatch for $tarball"; return 1; }

    tar -xJf "$workdir/$tarball" -C "$workdir" || { fail "failed to extract $tarball"; return 1; }

    local target="$TOOLS_DIR/node-$version-$platform"
    rm -rf "$target"
    mv "$workdir/node-$version-$platform" "$target"
    ln -sfn "$target" "$LOCAL_NODE_DIR"

    activate_local_node || { fail "installed Node.js is not executable at $LOCAL_NODE_DIR/bin/node"; return 1; }
    ok "Node.js ready: $(node --version) (npm $(npm --version 2>/dev/null || echo '?'))"
}

ensure_node() {
    step "Node.js toolchain"
    if node_is_supported; then
        ok "system Node.js $(node --version) is supported"
        return 0
    fi
    if activate_local_node && node_is_supported; then
        ok "using previously installed local Node.js $(node --version)"
        return 0
    fi
    warn "Node.js >= $NODE_MIN_MAJOR.x not found — installing a local copy under $LOCAL_NODE_DIR"
    install_node
}

# ---- Copy source & build ---------------------------------------------------------
ensure_python() {
    step "Checking Python (3.11 or newer)"
    local candidate
    if [[ -n "$PYTHON_BIN" ]]; then
        "$PYTHON_BIN" -c 'import sys, venv; sys.exit(sys.version_info < (3, 11))' ||
            { fail "PYTHON_BIN must be a working Python 3.11+ interpreter with venv."; exit 1; }
        return
    fi
    for candidate in python3 python3.12 python3.11; do
        if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys, ensurepip, venv; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
            PYTHON_BIN="$(command -v "$candidate")"
            return
        fi
    done
    info "Provisioning private Python 3.12 with uv $UV_VERSION (system Python is unchanged)."
    local bootstrap="$TOOLS_DIR/python-bootstrap"
    mkdir -p "$TOOLS_DIR"
    if [[ ! -x "$bootstrap/bin/python" ]]; then
        python3 -m venv "$bootstrap"
    fi
    "$bootstrap/bin/python" -m pip install --disable-pip-version-check "uv==$UV_VERSION"
    export UV_PYTHON_INSTALL_DIR="$TOOLS_DIR/python"
    export UV_PYTHON_BIN_DIR="$TOOLS_DIR/python-bin"
    "$bootstrap/bin/uv" python install 3.12
    PYTHON_BIN="$("$bootstrap/bin/uv" python find --managed-python 3.12)"
    "$PYTHON_BIN" -c 'import sys, ensurepip, venv; sys.exit(sys.version_info < (3, 11))'
}

sync_source() {
    step "Installing files to $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    if [[ "$(cd "$INSTALL_DIR" && pwd)" == "$SOURCE_DIR" ]]; then
        ok "installing in place ($SOURCE_DIR)"
        return 0
    fi
    rsync -a --delete \
        --exclude '.git' --exclude 'node_modules' --exclude '.venv' --exclude 'dist' \
        --exclude '.dev-pids' --exclude '.dev-logs' --exclude '.dev-tools' --exclude '.tools' \
        --exclude '.pytest_cache' --exclude 'session-artifacts' --exclude '.lens-pids' \
        --exclude '.lens-logs' --exclude 'private/run_store' --exclude 'certs' \
        "$SOURCE_DIR"/ "$INSTALL_DIR"/
    ok "copied source tree to $INSTALL_DIR"
}

ensure_venv() {
    step "Python virtual environment"
    cd "$INSTALL_DIR"

    # The embedded PostgreSQL mode needs the optional 'pgserver' extra and the
    # LDAP identity provider needs the optional 'ldap' extra; a plain
    # A plain backend install skips both. Install them so those features work.
    local extras=()
    [[ "$DB_MODE" == "embedded" ]] && extras+=("embedded-db")
    extras+=("ldap")
    local extra
    extra="[$(IFS=,; printf '%s' "${extras[*]}")]"

    # A stamp of apps/backend/pyproject.toml (plus the selected extra) forces a reinstall
    # when a dependency is added/bumped by a source update, e.g. casbin. Without
    # it, an existing .venv that still imports llm_d_lens/uvicorn/pgserver
    # would skip the install and the database bootstrap would fail on the new
    # module. sync_source excludes .venv, so the stamp survives re-installs.
    local stamp=".venv/.install-stamp"
    local current_hash=""
    if [[ -f apps/backend/pyproject.toml ]]; then
        current_hash="${extra}:$(sha256_file apps/backend/pyproject.toml 2>/dev/null || true)"
    fi

    local check_import="import llm_d_lens, uvicorn"
    [[ "$DB_MODE" == "embedded" ]] && check_import="$check_import, pgserver"
    if [[ -x .venv/bin/python && -f "$stamp" && "$(cat "$stamp" 2>/dev/null)" == "$current_hash" ]] \
        && .venv/bin/python -c "$check_import" 2>/dev/null; then
        ok "virtualenv already set up"
        return 0
    fi
    if [[ ! -x .venv/bin/python ]]; then
        info "creating virtualenv (.venv)..."
        "$PYTHON_BIN" -m venv .venv || { fail "Python virtualenv creation failed"; exit 1; }
    elif [[ -f "$stamp" && "$(cat "$stamp" 2>/dev/null)" != "$current_hash" ]]; then
        info "backend pyproject.toml changed since the last install — refreshing Python dependencies..."
    fi
    info "installing Python dependencies (pip install -e apps/backend${extra})..."
    .venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1 || true
    run_with_tail 5 .venv/bin/python -m pip install -e "apps/backend${extra}" || { fail "pip install -e apps/backend${extra} failed"; exit 1; }
    echo "$current_hash" > "$stamp"
    ok "Python dependencies installed"
}

# ---- Application database ("Step 0") ---------------------------------------------
# This used to be a mandatory gate inside CreateClusterWizard.jsx (blocking
# the wizard until configured). It now happens once here, before the backend
# is ever started, so the wizard can simply assume the database is ready —
# see llm_d_lens/db/bootstrap_cli.py.
configure_database() {
    step "Configuring the Lens database"
    cd "$INSTALL_DIR"
    local args=(--mode "$DB_MODE")
    if [[ "$DB_MODE" == "external" ]]; then
        if [[ -z "$DB_HOST" || -z "$DB_NAME" ]]; then
            fail "--db-mode external requires --db-host and --db-name (or run interactively)."
            exit 1
        fi
        args+=(--engine "$DB_ENGINE" --host "$DB_HOST" --dbname "$DB_NAME")
        [[ -n "$DB_PORT" ]] && args+=(--port "$DB_PORT")
        [[ -n "$DB_USER" ]] && args+=(--username "$DB_USER")
        [[ -n "$DB_PASSWORD" ]] && args+=(--password "$DB_PASSWORD")
    else
        if [[ -z "$DB_EMBEDDED_PASSWORD" ]]; then
            DB_EMBEDDED_PASSWORD="$(openssl rand -base64 24)"
            DB_EMBEDDED_PASSWORD_AUTOGEN=1
            info "generated a random embedded database password (stored in ~/.llm-d-lens/db-bootstrap.json)"
        fi
        args+=(--embedded-password "$DB_EMBEDDED_PASSWORD")
    fi
    local bootstrap_output
    if ! bootstrap_output="$(.venv/bin/python -m llm_d_lens.db.bootstrap_cli "${args[@]}" 2>&1)"; then
        printf '%s\n' "$bootstrap_output" >&2
        fail "database configuration failed — re-run with --db-mode embedded to use the built-in PostgreSQL, or fix the external database settings."
        exit 1
    fi
    if [[ "$bootstrap_output" == *"already configured -- skipping"* ]]; then
        # A database from a prior install/run was reused as-is; the
        # freshly generated password above was never applied, so don't
        # advertise it as this install's credential.
        DB_EMBEDDED_PASSWORD_AUTOGEN=0
        info "database already configured — reusing existing settings (use --force to reconfigure)"
    fi
    ok "database configured (mode: $DB_MODE)"
}

build_app() {
    step "Building the web app"
    cd "$INSTALL_DIR"
    info "running npm install ..."
    # --ignore-scripts: onnxruntime-node's postinstall tries to download a
    # native binary that's unreachable on some networks; harmless to skip
    # since the browser build (transformers.js) never uses it anyway.
    npm install --no-audit --no-fund --ignore-scripts || { fail "npm install failed"; exit 1; }
    info "running npm run build ..."
    npm run build || { fail "npm run build failed"; exit 1; }
    ok "build complete (apps/frontend/dist/)"
}

# ---- Config & control script ------------------------------------------------------
setup_tls() {
    [[ "$ENABLE_HTTPS" == "1" ]] || return 0
    step "Setting up HTTPS (self-signed certificate)"
    local host="$TLS_HOST"
    if [[ -z "$host" ]]; then
        host="$(hostos lan-ip)" || exit 1
    fi
    if [[ -z "$host" ]]; then
        warn "could not detect a LAN IP for the TLS certificate — falling back to plain HTTP."
        warn "re-run with --tls-host <your-ip> to enable HTTPS."
        ENABLE_HTTPS=0
        return 0
    fi
    local cert_env
    if ! cert_env="$("$INSTALL_DIR/hack/generate-self-signed-cert.sh" "$host" "$INSTALL_DIR/certs")"; then
        warn "failed to generate a self-signed TLS certificate — falling back to plain HTTP."
        ENABLE_HTTPS=0
        return 0
    fi
    eval "$cert_env"
    ok "certificate ready for $host ($INSTALL_DIR/certs)"
}

write_config() {
    step "Writing configuration"
    local in_place=0
    [[ "$(cd "$INSTALL_DIR" && pwd)" == "$SOURCE_DIR" ]] && in_place=1
    cat > "$INSTALL_DIR/.lens-install.env" <<EOF
# Generated by the Lens online installer — safe to edit and re-run lensctl.sh.
APP_PORT=$APP_PORT
BACKEND_PORT=$BACKEND_PORT
APP_HOST=0.0.0.0
LENS_IN_PLACE=$in_place
ENABLE_HTTPS=$ENABLE_HTTPS
TLS_CERT_FILE=${TLS_CERT_FILE:-}
TLS_KEY_FILE=${TLS_KEY_FILE:-}
DB_MODE=$DB_MODE
EOF
    ok "wrote $INSTALL_DIR/.lens-install.env"
}

write_control_script() {
    cat > "$INSTALL_DIR/lensctl.sh" <<'CTL'
#!/usr/bin/env bash
# lensctl.sh — start/stop/status for an installed llm-d Lens deployment.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
[[ -f .lens-install.env ]] && source .lens-install.env
APP_PORT="${APP_PORT:-3000}"
BACKEND_PORT="$APP_PORT"
APP_HOST="${APP_HOST:-0.0.0.0}"
LENS_IN_PLACE="${LENS_IN_PLACE:-0}"
ENABLE_HTTPS="${ENABLE_HTTPS:-1}"
TLS_CERT_FILE="${TLS_CERT_FILE:-}"
TLS_KEY_FILE="${TLS_KEY_FILE:-}"
DB_MODE="${DB_MODE:-embedded}"
PID_DIR="$ROOT_DIR/.prism-pids"
source "$ROOT_DIR/hack/storage-env.sh"
LOG_DIR="$LENS_LOG_DIR/service"
# Retained for operators/tools that inspect the generated script.
# shellcheck disable=SC2034
EVALUATE_STORE="$LENS_DATA_DIR/metadata/evaluations"

if [[ -x "$ROOT_DIR/.tools/node/bin/node" ]]; then
    PATH="$ROOT_DIR/.tools/node/bin:$PATH"
fi

log()  { printf '\033[1;36m[lens]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[lens]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[lens]\033[0m %s\n' "$*" >&2; }

# hostos — run the host OS layer with the installed virtualenv. It returns
# 0/1 to the caller and aborts on a usage/internal error (exit >= 2).
hostos_exec() {
    PYTHONPATH="$ROOT_DIR/apps/backend/src" \
        "$ROOT_DIR/.venv/bin/python" -m llm_d_lens.infrastructure.hostos "$@"
}
hostos() {
    local rc=0
    hostos_exec "$@" || rc=$?
    if (( rc >= 2 )); then
        fail "hostos command failed (exit $rc)"
        exit 1
    fi
    return "$rc"
}

pid_file() { printf '%s/%s.pid' "$PID_DIR" "$1"; }
is_running() {
    local pid; pid="$(cat "$(pid_file "$1")" 2>/dev/null || echo '')"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}
# Ports from .lens-install.env are hand-editable, so validate them before
# passing them to hostos (argparse would otherwise abort with a usage error).
valid_port() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 >= 1 && 10#$1 <= 65535 ))
}
# This host may be shared with other users/instances, so a previously chosen
# port can end up already bound by someone/something else by the time this
# runs again -- check right before starting rather than finding out from a
# crashed uvicorn/node log after the fact.
# Relies on an earlier `hostos check` gate: `port-in-use` exits 1 both when the port is free and when unsupported.
port_in_use() {
    hostos port-in-use -- "$1"
}
find_free_port() {
    hostos free-port -- "$1"
}
# describe_port_owner <port> -> best-effort "who's holding this port" string;
# on this kind of shared host the occupant is often another user's process,
# whose PID won't be visible without root, so say that plainly instead of
# guessing.
describe_port_owner() {
    hostos port-owner -- "$1"
}
# resolve_ports — auto-switch away from an already-occupied port instead of
# starting a service doomed to crash with EADDRINUSE; explains what's holding
# the old port (not just the new one picked) and reports the actual port
# used (also visible afterwards via .lens-install.env / status).
resolve_ports() {
    if ! valid_port "$APP_PORT"; then
        fail "APP_PORT must be a port between 1 and 65535 (got '$APP_PORT')."
        return 1
    fi
    local original owner
    if ! is_running "app" && port_in_use "$APP_PORT"; then
        original="$APP_PORT"
        owner="$(describe_port_owner "$original")" || exit 1
        warn "app port $original is already held by $owner."
        APP_PORT="$(find_free_port "$APP_PORT")" || exit 1
        warn "switched app port to $APP_PORT instead."
    fi
    BACKEND_PORT="$APP_PORT"
}
# Writes any port change resolve_ports made back to .lens-install.env, so it
# sticks across future stop/start/restart/status invocations instead of
# re-detecting (and re-warning about) the same conflict every time.
persist_ports() {
    [[ -f "$ROOT_DIR/.lens-install.env" ]] || return 0
    local tmp; tmp="$(mktemp "$ROOT_DIR/.lens-install.env.XXXXXX")"
    awk -v app="$APP_PORT" -v backend="$BACKEND_PORT" '
        /^APP_PORT=/     { print "APP_PORT=" app;     next }
        /^BACKEND_PORT=/ { print "BACKEND_PORT=" backend; next }
        { print }
    ' "$ROOT_DIR/.lens-install.env" > "$tmp"
    mv "$tmp" "$ROOT_DIR/.lens-install.env"
}
# Regenerates the self-signed cert if it's missing (e.g. certs/ was deleted),
# so a plain `lensctl.sh start` is always enough — no manual export needed.
ensure_tls() {
    [[ "$ENABLE_HTTPS" == "1" ]] || return 0
    if [[ -n "$TLS_CERT_FILE" && -n "$TLS_KEY_FILE" && -s "$TLS_CERT_FILE" && -s "$TLS_KEY_FILE" ]]; then
        return 0
    fi
    local host; host="$(hostos lan-ip)" || exit 1
    if [[ -z "$host" ]]; then
        warn "could not detect a LAN IP for the TLS certificate — running plain HTTP."
        ENABLE_HTTPS=0
        return 0
    fi
    local cert_env
    if ! cert_env="$("$ROOT_DIR/hack/generate-self-signed-cert.sh" "$host" "$ROOT_DIR/certs")"; then
        warn "failed to generate a self-signed TLS certificate — running plain HTTP."
        ENABLE_HTTPS=0
        return 0
    fi
    eval "$cert_env"
}
start_one() {
    local name="$1"; shift
    if is_running "$name"; then
        warn "$name already running (pid $(cat "$(pid_file "$name")"))"
        return 0
    fi
    mkdir -p "$PID_DIR" "$LOG_DIR"
    log "starting $name..."
    local run_log
    run_log="$name-$(date -u +%Y%m%dT%H%M%S%NZ)-$$.log"
    ln -sfn "$run_log" "$LOG_DIR/$name.log"
    nohup "$@" > "$LOG_DIR/$run_log" 2>&1 &
    echo "$!" > "$(pid_file "$name")"
}
wait_for_http() {
    local name="$1" url="$2" attempts="${3:-30}" i=0
    while [[ $i -lt $attempts ]]; do
        curl -sf -k -m 2 "$url" >/dev/null 2>&1 && return 0
        sleep 1; i=$((i + 1))
    done
    warn "$name did not become ready at $url — see $LOG_DIR/$name.log"
    return 1
}

# Points operators to the protected first-run credential file without
# printing its contents (mirrors hack/lens-local.sh's print_initial_admin_credentials).
# The file is removed once the password is changed.
print_initial_admin_credentials() {
    local cred_file="$LENS_DATA_DIR/credentials/initial_admin.txt"
    [[ -f "$cred_file" ]] || return 0
    log "administrator credentials are stored in the mode-600 file $cred_file; removed after the password is changed"
}

# Validate the Python catalog before starting the unified service.
regenerate_mcp_tools() {
    mkdir -p "$LOG_DIR"
    log "validating MCP tool catalog from the backend API..."
    if ! .venv/bin/python -c 'from llm_d_lens.api.main import app; from llm_d_lens.assistant_tools.catalog import catalog; print(f"Validated {len(catalog(app))} MCP tools")' >"$LOG_DIR/generate-mcp-tools.log" 2>&1; then
        fail "MCP tool catalog validation failed; see $LOG_DIR/generate-mcp-tools.log"
        return 1
    fi
}
cmd_start() {
    if ! hostos check >/dev/null; then
        fail "host check failed; this host is not supported."
        exit 1
    fi
    if is_running backend; then
        fail "The legacy backend is still running; use restart to switch to the unified service."
        return 1
    fi
    resolve_ports
    persist_ports
    ensure_tls
    # When the installed backend serves HTTPS with its self-signed certificate,
    # the Istio Gateway must originate unverified TLS to the lens-authz hop;
    # without it every Gateway request fails closed with 503. Mirrors
    # hack/lens-local.sh; an explicit environment value wins.
    local authz_tls_insecure="${LENS_MODEL_GATEWAY_AUTHZ_TLS_INSECURE:-}"
    if [[ -z "$authz_tls_insecure" ]]; then
        if [[ "$ENABLE_HTTPS" == "1" ]]; then authz_tls_insecure=1; else authz_tls_insecure=0; fi
    fi
    regenerate_mcp_tools
    local scheme="http"
    [[ "$ENABLE_HTTPS" == "1" ]] && scheme="https"
    start_one app env NODE_ENV=production BACKEND_HOST="$APP_HOST" BACKEND_PORT="$APP_PORT" \
        PRISM_MCP_URL="$scheme://127.0.0.1:$APP_PORT/api/assistant/mcp" \
        PRISM_MCP_CA_FILE="${TLS_CERT_FILE:-}" \
        LENS_MODEL_GATEWAY_AUTHZ_TLS_INSECURE="$authz_tls_insecure" \
        TLS_CERT_FILE="$TLS_CERT_FILE" TLS_KEY_FILE="$TLS_KEY_FILE" \
        bash hack/run-backend.sh
    log "waiting for services to become ready..."
    wait_for_http app     "$scheme://127.0.0.1:$APP_PORT/api/health"
    wait_for_http app     "$scheme://127.0.0.1:$APP_PORT/api/config"
    local lan_ip
    lan_ip="$(hostos lan-ip)" || exit 1
    log "llm-d Lens is running:"
    log "  app       $scheme://localhost:$APP_PORT  (bound to $APP_HOST)"
    [[ -n "$lan_ip" && "$APP_HOST" == "0.0.0.0" ]] && log "             $scheme://$lan_ip:$APP_PORT  (LAN)"
    if [[ "$ENABLE_HTTPS" == "1" ]]; then
        log "  HTTPS uses a self-signed cert — your browser will warn once; accept it to proceed. Needed for mic access etc. off localhost."
    fi
    print_initial_admin_credentials
}

# embedded_db_dir -> the on-disk location of the built-in PostgreSQL cluster
# (see llm_d_lens/db/settings.py's _DB_DATA_DIRECTORY_DEFAULT), honoring the
# same LLM_D_BENCH_EMBEDDED_DB_DIR override the backend itself respects.
embedded_db_dir() {
    local dir="${LLM_D_BENCH_EMBEDDED_DB_DIR:-~/.llm-d-lens/db/data}"
    printf '%s\n' "${dir/#\~/$HOME}"
}

# embedded_db_bootstrap_file -> the "Step 0" config recording how this install
# connects to its own database (see llm_d_lens/db/bootstrap_config.py),
# honoring the same LLM_D_BENCH_DB_BOOTSTRAP_FILE override the backend respects.
embedded_db_bootstrap_file() {
    local file="${LLM_D_BENCH_DB_BOOTSTRAP_FILE:-~/.llm-d-lens/db-bootstrap.json}"
    printf '%s\n' "${file/#\~/$HOME}"
}

# stop_embedded_postgres_process — best-effort graceful stop of the embedded
# PostgreSQL started by pgserver inside the backend process. Killing the
# backend's uvicorn pid does not guarantee its pgserver-managed postgres
# subprocess exits too, so read its postmaster.pid and signal it directly;
# otherwise it's left running as an orphan after `stop`/`uninstall`.
stop_embedded_postgres_process() {
    [[ "$DB_MODE" == "embedded" ]] || return 0
    local data_dir pid_file pid waited=0
    data_dir="$(embedded_db_dir)"
    pid_file="$data_dir/postmaster.pid"
    [[ -f "$pid_file" ]] || return 0
    pid="$(head -n1 "$pid_file" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 0
    log "stopping embedded PostgreSQL (pid $pid)..."
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

cmd_stop() {
    for svc in backend app; do
        local pf; pf="$(pid_file "$svc")"
        if [[ -f "$pf" ]]; then
            local pid; pid="$(cat "$pf")"
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                log "stopping $svc (pid $pid)..."
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pf"
        fi
    done
    stop_embedded_postgres_process
    log "stopped."
}

cmd_status() {
    local svc="app"
    if is_running "$svc"; then
        log "  $svc: running (pid $(cat "$(pid_file "$svc")"))"
    else
        log "  $svc: stopped"
    fi
    if is_running backend; then
        warn "legacy backend: running; use restart to switch to the unified service."
    fi
}

# cmd_uninstall — stop services and remove the installation. By default, user
# legacy repository-local data and the port config are kept by default.
# --purge removes the installation only; external LENS_* data roots are retained.
# The embedded database (if this install used --db-mode embedded) is asked
# about separately via --keep-db/--purge-db: its process is always stopped
# regardless of that answer, only its on-disk data is conditionally removed.
cmd_uninstall() {
    local mode="ask" db_mode="ask"
    for arg in "$@"; do
        case "$arg" in
            --keep-data) mode="keep" ;;
            --purge)     mode="purge" ;;
            --keep-db)   db_mode="keep" ;;
            --purge-db)  db_mode="purge" ;;
            -y|--yes)
                [[ "$mode" == "ask" ]] && mode="keep"
                [[ "$db_mode" == "ask" ]] && db_mode="keep"
                ;;
            *) fail "unknown uninstall option: $arg"; exit 1 ;;
        esac
    done

    if [[ "$mode" == "ask" ]]; then
        if [[ -t 0 ]]; then
            local reply
            read -r -p "Keep legacy installation data and port config? [Y/n] " reply || true
            reply="${reply:-Y}"
            [[ "$reply" =~ ^[Yy] ]] && mode="keep" || mode="purge"
        else
            warn "no TTY for prompt — defaulting to keep data (use --purge to remove it)."
            mode="keep"
        fi
    fi

    if [[ "$DB_MODE" == "embedded" && "$db_mode" == "ask" ]]; then
        if [[ -t 0 ]]; then
            local db_reply
            read -r -p "Delete the embedded database at $(embedded_db_dir)? [y/N] " db_reply || true
            db_reply="${db_reply:-N}"
            [[ "$db_reply" =~ ^[Yy] ]] && db_mode="purge" || db_mode="keep"
        else
            warn "no TTY for prompt — defaulting to keep the embedded database (use --purge-db to remove it)."
            db_mode="keep"
        fi
    fi

    log "External Lens data remains at $LENS_DATA_DIR (not removed by uninstall)."
    log "stopping services..."
    cmd_stop

    if [[ "$DB_MODE" == "embedded" ]]; then
        if [[ "$db_mode" == "purge" ]]; then
            log "deleting embedded database at $(embedded_db_dir)..."
            rm -rf "$(embedded_db_dir)"
            rm -f "$(embedded_db_bootstrap_file)"
            log "embedded database deleted."
        else
            log "embedded database stopped; data kept at $(embedded_db_dir)."
        fi
    fi

    if [[ "$LENS_IN_PLACE" == "1" ]]; then
        warn "this is an in-place install (running from the source checkout) —"
        warn "only build artifacts will be removed; source files are left untouched."
        rm -rf node_modules apps/frontend/dist docs/site/build .venv .tools "$PID_DIR"
        [[ "$mode" == "purge" ]] && rm -f .lens-install.env
        log "build artifacts removed from $ROOT_DIR."
        return 0
    fi

    if [[ "$mode" == "keep" ]]; then
        log "removing installed app, keeping private/run_store and .lens-install.env..."
        find "$ROOT_DIR" -mindepth 1 -maxdepth 1 \
            ! -name 'private' ! -name '.lens-install.env' -exec rm -rf {} +
        log "done. Data preserved at $ROOT_DIR/private and $ROOT_DIR/.lens-install.env"
        log "remove that directory manually if you no longer need the data."
    else
        log "purging everything, including private/run_store..."
        cd /
        rm -rf "$ROOT_DIR"
        log "done. $ROOT_DIR has been removed."
    fi
}

case "${1:-}" in
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_stop; cmd_start ;;
    status)    cmd_status ;;
    uninstall) shift; cmd_uninstall "$@" ;;
    *) echo "Usage: $0 {start|stop|restart|status|uninstall [--keep-data|--purge|-y]}"; exit 1 ;;
esac
CTL
    chmod +x "$INSTALL_DIR/lensctl.sh"
    ok "wrote $INSTALL_DIR/lensctl.sh"
}

# ---- Main -------------------------------------------------------------------------
run_uninstall() {
    print_logo
    if [[ "$SILENT" != "1" ]]; then
        INSTALL_DIR="$(ask "Directory to uninstall from" "$INSTALL_DIR")"
    fi
    if [[ ! -x "$INSTALL_DIR/lensctl.sh" ]]; then
        fail "no llm-d Lens installation found at $INSTALL_DIR (lensctl.sh missing)."
        exit 1
    fi
    if [[ "$SILENT" != "1" && ${#UNINSTALL_ARGS[@]} -eq 0 ]]; then
        if ! confirm "Uninstall llm-d Lens from $INSTALL_DIR" "N"; then
            warn "aborted by user."
            exit 1
        fi
    fi
    "$INSTALL_DIR/lensctl.sh" uninstall "${UNINSTALL_ARGS[@]}"
}

install_from_source() {
    if [[ "$UNINSTALL" == "1" ]]; then
        run_uninstall
        return 0
    fi

    print_logo
    require_host_os
    if [[ "$FRESH_INSTALL" == 1 ]]; then
        if [[ -L "$INSTALL_DIR" ]] || { [[ -e "$INSTALL_DIR" ]] &&
            { [[ ! -d "$INSTALL_DIR" ]] || [[ -n "$(find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; }; then
            fail "Fresh installation requires a new or empty directory: $INSTALL_DIR"
            exit 1
        fi
    fi

    if [[ "$SILENT" != "1" ]]; then
        interactive_configure
    else
        resolve_ports_silently
        info "silent mode — install dir: $INSTALL_DIR, app/API port: $APP_PORT"
    fi

    ensure_system_packages
    TOOLS_DIR="$INSTALL_DIR/.tools"
    LOCAL_NODE_DIR="$TOOLS_DIR/node"
    ensure_node
    ensure_python
    sync_source
    ensure_venv
    configure_database
    build_app
    setup_tls
    write_config
    write_control_script

    step "Done"
    ok "llm-d Lens installed at $INSTALL_DIR"
    if [[ "$DB_MODE" == "embedded" && "$DB_EMBEDDED_PASSWORD_AUTOGEN" == "1" ]]; then
        printf '%s[lens]%s %s⚠%s auto-generated embedded database password (save this — it will not be shown again):\n' \
            "$YELLOW$BOLD" "$RESET" "$YELLOW" "$RESET"
        printf '    %s%s%s\n' "$BOLD" "$DB_EMBEDDED_PASSWORD" "$RESET"
        info "also recoverable later from ~/.llm-d-lens/db-bootstrap.json on this machine."
    fi
    info "manage it with:"
    printf '    %s%s/lensctl.sh start|stop|restart|status|uninstall%s\n' "$BOLD" "$INSTALL_DIR" "$RESET"

    if [[ "$START_AFTER_INSTALL" == "1" ]]; then
        step "Starting llm-d Lens"
        "$INSTALL_DIR/lensctl.sh" start
    else
        info "not starting automatically — run '$INSTALL_DIR/lensctl.sh start' when ready."
    fi
}

release_version() {
    [[ $# == 2 ]] || online_fail "Usage: release stamp|check main|vMAJOR.MINOR.PATCH"
    [[ "$PIPE_ENTRY" == 0 ]] || online_fail "Release maintenance requires a local script file."
    local action="$1" version="$2" script="${BASH_SOURCE[0]}" embedded
    [[ "$action" == stamp || "$action" == check ]] || online_fail "Expected release stamp or release check."
    [[ "$version" == main || "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]] ||
        online_fail "Expected main or a vMAJOR.MINOR.PATCH tag (optional prerelease)."
    [[ ! -L "$script" && "$(grep -c '^LENS_SOURCE_VERSION="[^"]*"$' "$script")" == 1 ]] ||
        online_fail "Expected a regular script with exactly one source-version assignment."
    embedded="$(sed -n 's/^LENS_SOURCE_VERSION="\([^"]*\)"$/\1/p' "$script")"
    if [[ "$action" == stamp ]]; then
        sed -i "s/^LENS_SOURCE_VERSION=\"[^\"]*\"$/LENS_SOURCE_VERSION=\"$version\"/" "$script"
        printf '[lens-online] Stamped source version: %s\n' "$version"
    else
        [[ "$embedded" == "$version" ]] ||
            online_fail "Embedded version $embedded does not match $version."
        printf '[lens-online] Source version verified: %s\n' "$version"
    fi
}

if [[ "$PIPE_ENTRY" == 1 || "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        release)
            shift
            release_version "$@"
            ;;
        --install-source)
            shift
            [[ "$PIPE_ENTRY" == 0 && -f "$SOURCE_DIR/package.json" && -f "$SOURCE_DIR/apps/backend/pyproject.toml" &&
                -f "$SOURCE_DIR/hack/storage-env.sh" ]] ||
                online_fail "Internal source installation requires a complete downloaded Lens tree."
            parse_args "$@"
            FRESH_INSTALL=1
            [[ "$UNINSTALL" == 0 ]] || online_fail "Use the installed lensctl.sh to uninstall."
            install_from_source
            ;;
        *)
            # Keep terminal redirection out of the Bash reader consuming the pipe.
            (online_main "$@")
            ;;
    esac
fi
}
