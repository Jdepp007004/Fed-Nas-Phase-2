# Security Policy — FL Platform

## Supported Versions

| Version  | Supported          |
|----------|--------------------|
| ≥ 2.0    | ✅ Active security patches |
| 1.x      | ⚠️ Critical fixes only (EOL: 2026-12-31) |
| < 1.0    | ❌ Unsupported |

---

## Reporting a Vulnerability

> **Do NOT open a public GitHub issue for security vulnerabilities.**

### Disclosure Process

1. **Email** your report to: `security@fl-platform.internal`  
   (Substitute your organisation's actual security alias.)
2. **PGP encrypt** your message if the report contains exploit code or PHI.
   Public key fingerprint: `(set before production deployment)`
3. **Include** in your report:
   - Affected component(s) and version
   - Reproduction steps (minimal PoC preferred)
   - Impact assessment (data exposure, privilege escalation, etc.)
   - Whether you believe PHI / clinical data is at risk

### Response SLA

| Milestone | Target |
|-----------|--------|
| Acknowledgement | **24 hours** |
| Initial triage + severity classification | **72 hours** |
| Fix published (Critical/High) | **14 days** |
| Fix published (Medium/Low) | **90 days** |
| Public disclosure (coordinated) | **90 days after fix** |

We follow **coordinated responsible disclosure**. If you have a deadline
requirement, please state it in the initial report so we can negotiate.

---

## Threat Model

The FL Platform processes clinical data from multiple participating hospitals.
The key adversarial scenarios in scope are:

| Threat | Mitigation |
|--------|-----------|
| Malicious FL client poisoning the global model | Coordinate-wise trimmed mean (`trimming_ratio`), FLAME clustering defense, per-client reputation scores |
| Gradient inversion / model inversion attack | AES-256-GCM weight encryption + Gaussian DP noise (`dp_epsilon`) |
| Unauthorised access to clinical weights in transit | Mutual TLS (`MTLS_*` env vars), JWT auth on all API endpoints |
| Server compromise — exfiltration of aggregated model | Secure aggregation (masked FedAvg), model stored encrypted at rest via S3 SSE-KMS |
| Privacy leakage from released model | Differential privacy budget tracking (`PrivacyAccountant`); hard stop when budget exceeded |
| Insider threat (rogue admin) | Admin actions require `X-Admin-Key` header (separate from JWT); all actions logged |
| HIPAA BAA non-compliance | `baa_signed` flag gate on all project-participation endpoints |

---

## Security Controls Summary

### Authentication & Authorisation
- JWT HS256 tokens (configurable secret, 24h expiry)
- bcrypt password hashing (cost factor 12)
- Role-based access: `client`, `admin`
- Admin actions require a second credential (`X-Admin-Key`)

### Data in Transit
- All API endpoints should be exposed behind HTTPS (reverse proxy or `MTLS_*`)
- Weight updates encrypted with AES-256-GCM before transmission
- Optional mutual TLS for client certificate pinning

### Data at Rest
- Model checkpoints stored as `.pt` files; use S3 SSE-KMS or encrypted EBS
- JSON flat-file DB (`database.json`) — production should migrate to PostgreSQL with TDE

### Differential Privacy
- Gaussian mechanism applied to client weight updates before transmission
- Centralised `PrivacyAccountant` tracks (ε, δ) consumption; warns at budget exhaustion
- See `.env.example` for `DP_EPSILON`, `DP_DELTA`, `DP_MAX_GRAD_NORM`

### Compliance
- HIPAA BAA enforcement via `baa_signed` user flag (Phase 3)
- Per-hospital, per-project consent records (Phase 3)
- Right-to-deletion / federated unlearning (Phase 3)
- Data residency: pending updates purged from server memory after each round

---

## Incident Response Runbook

### Step 1 — Detection
- Prometheus alert fires on anomalous round metrics (val_rmse spike, round duration > 2×P95)
- Suspicious client IP detected in auth logs
- `AGGREGATING` round state stuck > `ROUND_STALL_TIMEOUT_S`

### Step 2 — Containment
```bash
# Immediately revoke the suspect client
curl -X POST http://server:8000/api/projects/{proj_id}/kick-client \
  -H "X-Admin-Key: $JWT_SECRET" \
  -d '{"user_id": "suspect-hospital-uuid"}'

# Reset stuck round
curl -X POST http://server:8000/api/projects/{proj_id}/reset-round \
  -H "X-Admin-Key: $JWT_SECRET"

# If server compromise suspected — bring down the fleet
docker compose down fl-server fl-worker fl-beat
```

### Step 3 — Eradication
- Roll back global model to last known-good checkpoint in `server/models/`
- Identify which rounds the suspect client participated in
- Run federated unlearning for those rounds (see `server/unlearning.py`)

### Step 4 — Recovery
- Restart services with rotated `JWT_SECRET` and `FL_ENCRYPTION_KEY`
- Increment Alembic migration if schema was altered
- Re-invite hospitals with fresh approval tokens

### Step 5 — Post-Incident Report
- Document: affected rounds, clients, metrics delta, timeline
- File HIPAA breach notification within **60 days** if PHI was exposed
- Update threat model and incident runbook

---

## Known Limitations

- The JSON flat-file DB (`database.json`) is not encrypted at rest — use PostgreSQL + TDE in production
- The in-process `RedisState` memory fallback is not persistent across restarts
- Secure aggregation (Phase 3.1) requires all clients to be online simultaneously for the setup phase

---

## Acknowledgements

We thank the security researchers who have responsibly disclosed issues to us.
Researchers are acknowledged in release notes with their consent.
