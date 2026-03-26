# Security Policy

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in ChaosBench-Logic, please report it responsibly.

### How to Report

**Preferred Method:** Create a GitHub Security Advisory at:
- https://github.com/11NOel11/ChaosBench-Logic/security/advisories/new

**What to Include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What NOT to Disclose Publicly

**Please do NOT:**
- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed
- Exploit the vulnerability beyond what is necessary to demonstrate it

### Response Timeline

- **Initial Response:** Best effort within 5 business days
- **Status Update:** Within 10 business days of initial report
- **Resolution:** Depends on severity and complexity

### Scope

This security policy applies to:
- Core evaluation logic (`chaosbench/eval/`, `chaosbench/models/adapters/`)
- Dataset processing and validation scripts
- CI/CD workflows

**Out of Scope:**
- Issues in third-party dependencies (report to upstream projects)
- General bug reports (use GitHub Issues)
- Feature requests

### Supported Versions

We provide security updates for:
- Latest release on the `master` branch
- No backporting to older versions

### Security Best Practices

When using ChaosBench-Logic:
- **Never commit API keys** - Use `.env` files (gitignored) or environment variables
- **Validate external model outputs** - The benchmark evaluates untrusted LLM responses
- **Review dataset modifications** - Ensure data integrity if modifying benchmark data

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who report valid vulnerabilities (with permission).
