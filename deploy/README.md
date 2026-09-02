# Deploying the footfall pipeline

Two supervised options. Both keep the process running across crashes and
reboots and emit line-delimited JSON logs.

## Container (recommended for a single box)

```bash
# 1. put the site config in place (not baked into the image)
cp config/site.example.yaml config/site.yaml
cp config/secrets.example.yaml config/secrets.yaml   # then edit both

# 2. build + run, restarts automatically
docker compose up -d --build

# 3. logs
docker compose logs -f
```

`config/` is mounted read-only; `models/` (downloaded YOLO weights) and
`output/` (the SQLite event store and recordings) are named volumes, so
they survive `docker compose down`.

For a USB camera uncomment the `devices:` block; for LAN RTSP cameras you
usually need `network_mode: host`. For an NVIDIA box, base the image on
`nvidia/cuda`, install the CUDA torch wheel, and add `--gpus all`.

## systemd (bare metal)

See the header of [`footfall.service`](footfall.service) for the install
steps. In short: a `footfall` system user, a venv under `/opt/footfall`,
the unit file copied to `/etc/systemd/system/`, then
`systemctl enable --now footfall`. Logs go to the journal
(`journalctl -u footfall -f`).

## Configuration

| Variable | Meaning |
|---|---|
| `FOOTFALL_LOG_FORMAT` | `json` (default off a TTY) or `text` |
| `FOOTFALL_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` … |
| `FOOTFALL_SECRET_<NAME>` | a `${SECRET:name}` value for a camera source |
| `FOOTFALL_SECRETS_FILE` | path to the secrets YAML (default `config/secrets.yaml`) |

Managing the event store: `python tools/events_db.py runs` /
`drop-run <id>` / `reset`.
