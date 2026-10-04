# Shared checks for the git hooks in this directory (sourced, not executed). See README, "Data policy".
#
# Blocks: data / model / CAD / CFD / office files by extension, files larger than MAX_KB, files under
# restricted directory names, credentials, and the project-specific sensitive terms listed (one per line,
# case-insensitive) in $(git rev-parse --git-dir)/sensitive-terms. That list is deliberately kept inside .git/,
# so it is never committed or pushed: the terms themselves would reveal what must stay confidential.
# Intentional exceptions: add a path glob to .githooks/allowlist (reviewed like any other change).

MAX_KB=2048
BLOCKED_EXT='\.(h5|hdf5|he5|nc|cdf|npy|npz|mat|pt|pth|ckpt|safetensors|onnx|bin|arrow|parquet|feather|pkl|pickle|joblib|tfrecord|zarr|vtk|vtu|vtp|vts|pvd|xdmf|xmf|cgns|cas|msh|foam|stl|obj|ply|step|stp|iges|igs|x_t|sldprt|sldasm|dwg|dxf|xlsx|xls|docx|doc|pptx|ppt|pdf|zip|tar|gz|tgz|bz2|xz|7z|rar)$'
BLOCKED_DIR='(^|/)(confidential|restricted|private|partner|partner_data|raw_data)(/|$)|^data/'
SECRET_RE='hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}'

TOP=$(git rev-parse --show-toplevel)
TERMS_FILE="$(git rev-parse --git-dir)/sensitive-terms"
ALLOW_FILE="$TOP/.githooks/allowlist"

say() { printf '\033[31m[data-policy]\033[0m %s\n' "$*" >&2; }

allowed() {  # $1 = path
  [ -f "$ALLOW_FILE" ] || return 1
  local g
  while IFS= read -r g; do
    [ -z "$g" ] || [ "${g#\#}" != "$g" ] && continue
    # shellcheck disable=SC2053
    [[ "$1" == $g ]] && return 0
  done < "$ALLOW_FILE"
  return 1
}

terms() {  # non-empty, non-comment lines of the local sensitive-terms file
  [ -f "$TERMS_FILE" ] && grep -v -e '^[[:space:]]*$' -e '^[[:space:]]*#' "$TERMS_FILE"
}

check_path() {  # $1 = path, $2 = size in bytes (optional)
  local p=$1 size=${2:-} bad=0 t
  allowed "$p" && return 0
  if printf '%s' "$p" | grep -qiE "$BLOCKED_EXT"; then say "blocked file type: $p"; bad=1; fi
  if printf '%s' "$p" | grep -qiE "$BLOCKED_DIR"; then say "blocked directory: $p"; bad=1; fi
  if [ -n "$size" ] && [ "$size" -gt $((MAX_KB * 1024)) ]; then say "file larger than ${MAX_KB} KB: $p ($((size / 1024)) KB)"; bad=1; fi
  while IFS= read -r t; do
    if printf '%s' "$p" | grep -qiF -- "$t"; then say "sensitive term in path: $p"; bad=1; fi
  done < <(terms)
  return $bad
}

check_text() {  # stdin = text to scan (added diff lines or a commit message); $1 = label for messages
  local label=$1 text bad=0 t
  text=$(cat)
  if printf '%s' "$text" | grep -qE -- "$SECRET_RE"; then say "credential-like string in $label"; bad=1; fi
  while IFS= read -r t; do
    if printf '%s' "$text" | grep -qiF -- "$t"; then say "sensitive term in $label (see .git/sensitive-terms)"; bad=1; fi
  done < <(terms)
  return $bad
}

added_lines() {  # stdin = unified diff; prints only added lines, skipping files on the allowlist
  local f=""
  while IFS= read -r line; do
    case "$line" in
      "+++ b/"*) f=${line#+++ b/} ;;
      "+++ "*) f="" ;;
      "+"*) allowed "$f" || printf '%s\n' "${line#+}" ;;
    esac
  done
}

blocked_msg() {
  say "Commit/push refused. This repository must never contain partner or confidential data,"
  say "or anything derived from it (see README, 'Data policy'). Fix the files above; do not bypass with --no-verify."
}
