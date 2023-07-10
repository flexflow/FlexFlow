#! /usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 <input-file.puml>"
}

print_help() {
cat <<USAGE
A hacky auto-refresh script for plantuml. There's probably something better out
there, but I was too lazy to find it.

Usage: 
  $0 <input-file.puml>

Note: you will need xdotool, plantuml, and inotify-tools installed. Currently
      this only works with firefox, but I'm sure you can figure out how to
      modify it.

Based on https://stackoverflow.com/a/66239637. Contact @lockshaw if you have
any questions.
USAGE
}

if [[ ! $# -eq 1 ]]; then 
  1>&2 usage
  exit 1
fi

if [[ $1 == "-h" || $1 == "--help" ]]; then 
  1>&2 print_help
  exit 0
fi

FILE="$1"
FILE="${FILE%.svg}"
FILE="${FILE%.puml}"

if [[ ! -f $FILE.puml ]]; then
  1>&2 echo "Could not find file $FILE.puml. Exiting..."
  exit 1
fi

set -x

TERM_WINDOWID="$(xdotool getactivewindow)"
plantuml -tsvg "$FILE.puml" || true
firefox "$FILE.svg"
FF_WINDOWID="$(xdotool search --name "Mozilla Firefox")"
xdotool windowactivate "$TERM_WINDOWID"
while [[ -f $FILE.puml ]]; do 
  inotifywait -e close_write "$FILE.puml" || true
  plantuml -tsvg "$FILE.puml" || true
  xdotool windowactivate "$FF_WINDOWID"
  xdotool key 'CTRL+r'
  xdotool windowactivate "$TERM_WINDOWID"
done
