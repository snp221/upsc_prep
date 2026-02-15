#!/bin/bash
# Auto-create .env from Codespace secrets

cat > .env << EOF
OPENAI_API_KEY=${OPENAI_API_KEY}
GOOGLE_SERVICE_ACCOUNT_JSON=${GOOGLE_SERVICE_ACCOUNT_JSON}
DEBUG=False
SECRET_KEY=${SECRET_KEY:-django-insecure-codespace-key}
EOF

echo ".env file created from Codespace secrets"
