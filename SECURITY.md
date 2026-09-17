# Security Policy

## Supported versions

The latest main branch is supported with security fixes.

## Reporting

Please do not open a public issue for a vulnerability. Send a private report to the repository maintainers with:

- affected version or commit
- reproduction steps
- impact assessment
- suggested mitigation, if available

## Deployment notes

- Keep DASHSCOPE_API_KEY and RKB_API_KEY in environment variables or a secret manager.
- Do not commit .env files.
- Set RKB_API_KEY when exposing the API beyond localhost.
- Restrict CORS origins in production.
- Treat uploaded documents as untrusted input.
- Back up data/index and data/uploads together.
