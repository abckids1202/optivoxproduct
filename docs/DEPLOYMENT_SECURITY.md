# OptiVox Deployment Security Runbook

This runbook applies to the canonical edge runtime in `main.py`, the FastAPI
backend in `backend/`, and the React application in `frontend/`. The
`optivox-web/` tree is retained legacy/marketing material and is not a second
production runtime.

## Runtime modes

Use `development` for local engineering and `exhibition` for a controlled
local demonstration. These modes retain loopback convenience, but they are
not suitable for an exposed service. Use `pilot` or `production` only after
setting authentication, an encrypted biometric key, explicit frontend origin,
trusted model files, and the required checksum manifests.

Strict startup fails closed when:

- a configured model is missing, unlisted, or has a different SHA-256 checksum;
- the local InsightFace `buffalo_l` cache is absent or unverified;
- a camera source uses an unsafe scheme, escapes the project root, or is
  missing when a local file source is configured;
- runtime mode, camera IDs, dimensions, or confidence values are invalid;
- configured webhook destinations are not HTTPS and allowlisted.

## Supply chain procedure

1. Create a clean virtual environment using the pinned `requirements.txt` or
   the reviewed `requirements.lock.txt`.
2. Run `python -m pip check` and resolve every conflict before promotion.
3. Run `python supply_chain_scan.py` and `python security_scan.py`.
4. Generate `python generate_sbom.py`; review `reports/sbom.json`.
5. Obtain model artifacts from a trusted, licensed source and inspect them
   before placing them on the edge machine.
6. Generate the project manifest with `python generate_model_manifest.py`.
7. Generate the InsightFace manifest inside its cache, for example:
   `python generate_model_manifest.py --root "%USERPROFILE%\.insightface\models\buffalo_l" --output model_checksums.json`.
8. Preserve the manifests with the deployment record. Do not commit weights,
   secrets, biometric databases, evidence, or generated runtime files.

The lock files pin versions but do not prove package safety. The inherited
development environment may differ from the reviewed target; `pip check` is a
promotion gate, not an optional warning.

## Network and secrets

Keep the edge runtime bound to loopback unless a trusted reverse proxy is
configured. Use HttpOnly authenticated sessions in pilot/production, TLS at
the proxy, strict CORS, and no frontend API-key variables. Supply SMTP,
Telegram, and webhook credentials through environment or an OS secret store.
Set `OPTIVOX_ALLOWED_WEBHOOK_HOSTS` to the exact approved host suffixes.
Webhook query strings, fragments, embedded credentials, redirects, arbitrary
schemes, and private/loopback production destinations are rejected.

The edge should have outbound access only to approved alert and update
destinations. It should not have unrestricted administrative network access.
Do not expose the FastAPI debug/docs surface publicly without an authenticated
network boundary.

## Windows least privilege

Run the engine and backend under a dedicated standard user or service account.
Grant it camera-device access, read access to trusted model/cache locations,
and read/write access only to the database, runtime, snapshot, report, export,
backup, and log directories. Keep `.env`, alert credentials, biometric keys,
and backups ACL-restricted to that account and administrators. Do not run the
camera service from an administrator shell or a developer profile.

## Health and recovery

`runtime/heartbeat.json` contains process identity, watchdog status, camera
state, and frame age. `runtime/health_events.jsonl` records camera transitions
such as `HEALTHY`, `NO_FRAME`, `FROZEN`, and `DISCONNECTED`; the backend health
route exposes a bounded recent view. Configure an external process supervisor
to restart the service only after capturing the health event and last error.

Back up the SQLite database through the protected System action, retain the
checksum manifest with the backup, and test restore into an isolated temporary
directory. Treat an audit-chain integrity failure, biometric checksum failure,
or evidence checksum failure as an investigation event rather than silently
repairing the file.

## Promotion checklist

- `OPTIVOX_RUNTIME_MODE` is explicitly set and is not `development`.
- `python -m pip check`, unit tests, compilation, build, secret scan, and
  supply-chain scan pass.
- Project and InsightFace model manifests verify successfully.
- No demo mode, loopback bypass, wildcard CORS, or public raw biometric route
  is enabled.
- Camera disconnect and freeze events are visible in health telemetry.
- A backup and isolated restore have been tested.
- A fresh ten-minute camera benchmark is attached to the deployment record.
- The operator knows how to stop the runtime safely and investigate alerts.
