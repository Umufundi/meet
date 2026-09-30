#!/usr/bin/env sh
# Install Meet on macOS or Linux: core app, `meet` on PATH, then `meet setup`.
#
#   sh scripts/install.sh [--skip-setup]
#
# Mirrors scripts/install.ps1. Everything lands under ~/.meet (or $MEET_HOME);
# nothing is installed system-wide.
set -eu

SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
MEET_HOME="${MEET_HOME:-$HOME/.meet}"
CORE="$MEET_HOME/core"
BIN="$MEET_HOME/bin"

fail() { printf '  x %s\n      %s\n' "$1" "$2" >&2; exit 1; }

echo "MEET INSTALL"
PYTHON=""
for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] not in ((3, 12), (3, 13)))' 2>/dev/null; then
        PYTHON="$candidate"; break
    fi
done
[ -n "$PYTHON" ] || fail "Python 3.12 or 3.13 was not found" "Install it (macOS: brew install python@3.12) and re-run."
echo "  + $($PYTHON --version)"

mkdir -p "$MEET_HOME" "$BIN"
[ -x "$CORE/bin/python" ] || "$PYTHON" -m venv "$CORE" || fail "could not create $CORE" \
    "Debian/Ubuntu: sudo apt install python3.12-venv"
PIP="$CORE/bin/python -m pip --disable-pip-version-check install --no-input --quiet"
$PIP --require-hashes --no-deps -r "$SOURCE/requirements.lock" || fail "installing dependencies failed" "Check your connection."
$PIP --no-deps --force-reinstall "$SOURCE" || fail "installing Meet failed" "See the error above."
rm -f "$BIN/meet"
printf '#!/bin/sh\nexec "%s/bin/python" -Pm meet "$@"\n' "$CORE" > "$BIN/meet"
chmod +x "$BIN/meet"
echo "  + Meet app in $CORE"

case ":$PATH:" in
    *":$BIN:"*) ;;
    *) echo "  ! add $BIN to your PATH, e.g.: echo 'export PATH=\"$BIN:\$PATH\"' >> ~/.profile" ;;
esac

[ "${1:-}" = "--skip-setup" ] && { echo "  done. Next: meet setup"; exit 0; }
exec "$BIN/meet" setup
