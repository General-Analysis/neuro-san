#!/bin/bash

# Usage: ./convert_yaml_to_json.sh input.yaml output.json

INPUT="$1"
OUTPUT="$2"

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  echo "❗ Usage: $0 input.yaml output.json"
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "❌ File not found: $INPUT"
  exit 1
fi

echo "🔄 Converting $INPUT → $OUTPUT..."

if command -v yq &>/dev/null; then
  echo "✅ Using yq..."
  yq -o=json "$INPUT" > "$OUTPUT"
elif command -v python3 &>/dev/null; then
  echo "⚠️ yq not found — falling back to Python..."
  python3 - <<EOF
import sys, json, yaml
with open("$INPUT") as f:
    data = yaml.safe_load(f)
with open("$OUTPUT", "w") as f:
    json.dump(data, f, indent=2)
EOF
else
  echo "❌ Neither 'yq' nor 'python3' is available. Please install one of them."
  exit 1
fi

echo "✅ Done! Output saved to $OUTPUT"
