#!/bin/bash
set -euo pipefail

cat > /app/web_status.txt <<'EOF'
web-search: rejected
web-fetch: rejected
EOF
