# DeepSeek Python API

Standalone OpenAI-compatible Python server for DeepSeek's private web chat API, with text, search/thinking aliases, streaming, and image upload support.

> This adapter is unofficial. It uses a DeepSeek browser/web authentication token and can break if DeepSeek changes its private web protocol.

## Features

- `POST /v1/chat/completions` with OpenAI-style request bodies.
- Streaming and non-streaming responses.
- OpenAI-style multimodal `image_url` parts, including base64 data URLs and remote HTTP(S) images.
- DeepSeek web-token exchange, per-request upstream session creation, best-effort session deletion, target-specific proof-of-work, image upload, file-status polling, and completion submission.
- Managed multi-token store with CRUD APIs, redacted token views, health checks, cooldowns, fill-first routing, and round-robin routing.
- Dependency-free dashboard at `GET /dashboard` for token management, health checks, and rotation changes.
- Optional local client-facing `API_KEY` separate from the DeepSeek upstream token.
- Optional management-facing `MANAGEMENT_API_KEY` for `/v0/management/*`.
- Optional trusted-mode `X-DeepSeek-Token` request header when `ALLOW_REQUEST_TOKEN=true`.
- Remote-image SSRF protections enabled by default.

## Quick start

From this folder:

```bash
/usr/bin/python3 -m venv .venv
```

```bash
.venv/bin/pip install -e '.[dev]'
```

```bash
cp .env.example .env
```

Set either `DEEPSEEK_AUTH_TOKEN` or managed tokens through the dashboard/API. For a managed-token deployment, set `MANAGEMENT_API_KEY` first. Then run:

```bash
.venv/bin/deepseek-python-api
```

The service listens on `http://127.0.0.1:8000` by default.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `DEEPSEEK_AUTH_TOKEN` | unset | Static upstream DeepSeek browser/web token. Used only when no managed tokens are configured. |
| `API_KEY` | unset | Optional client-facing API key for `/v1/*`. Do not reuse the DeepSeek token here. |
| `MANAGEMENT_API_KEY` | unset | Optional management key for `/v0/management/*`; falls back to `API_KEY` if unset. |
| `TOKENS_FILE` | `deepseek_tokens.json` | Local JSON store for managed DeepSeek tokens and rotation state. |
| `ROTATION_STRATEGY` | `fill_first` | Managed-token strategy: `fill_first` or `round_robin`. |
| `TOKEN_HEALTH_CHECK_INTERVAL_SECONDS` | `300` | Periodic health-check interval; set `0` to disable the background supervisor. |
| `TOKEN_FAILURE_COOLDOWN_SECONDS` | `300` | Cooldown duration after a non-auth token failure. |
| `ALLOW_REQUEST_TOKEN` | `false` | Allows trusted clients to send `X-DeepSeek-Token` per request. |
| `HOST` | `127.0.0.1` | Uvicorn bind host. |
| `PORT` | `8000` | Uvicorn bind port. |
| `LOG_LEVEL` | `INFO` | Uvicorn log level. |
| `MAX_IMAGE_BYTES` | `104857600` | Maximum local/remote image size. |
| `MAX_IMAGES` | `10` | Maximum image parts per request. |
| `ALLOW_PRIVATE_IMAGE_URLS` | `false` | Allows private/reserved remote-image hosts when explicitly enabled. |
| `DELETE_SESSIONS` | `true` | Best-effort upstream session cleanup after completion/disconnect. |

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /dashboard`
- `GET /v0/management/status`
- `GET /v0/management/tokens`
- `POST /v0/management/tokens`
- `GET /v0/management/tokens/{token_id}`
- `PATCH /v0/management/tokens/{token_id}`
- `DELETE /v0/management/tokens/{token_id}`
- `POST /v0/management/tokens/{token_id}/check`
- `POST /v0/management/tokens/check`
- `PATCH /v0/management/rotation`

If `API_KEY` is set, `/v1/*` clients must send either:

```text
Authorization: Bearer <API_KEY>
```

or:

```text
X-API-Key: <API_KEY>
```

Management endpoints require `MANAGEMENT_API_KEY`, or `API_KEY` when no dedicated management key is set. The dashboard HTML is static and does not embed secrets; it stores the management key only in the browser's localStorage and sends it to the protected management endpoints.

## Multi-token management

Managed tokens are persisted in `TOKENS_FILE` because the upstream web token must be replayed to DeepSeek. API responses and the dashboard never return raw token values; they expose only a short redaction and a SHA-256 fingerprint prefix. Keep the token store private, backed up carefully if needed, and out of git. The default local `deepseek_tokens.json` path is ignored by `.gitignore` and `.dockerignore`; the Docker image defaults to `/data/deepseek_tokens.json` so it can run as a non-root user with a mounted volume.

Add a token:

```bash
curl -sS http://127.0.0.1:8000/v0/management/tokens \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <MANAGEMENT_API_KEY>' \
  -d '{"name":"primary","token":"<deepseek-web-token>","check":true}'
```

List tokens:

```bash
curl -sS http://127.0.0.1:8000/v0/management/tokens \
  -H 'Authorization: Bearer <MANAGEMENT_API_KEY>'
```

Change rotation strategy:

```bash
curl -sS -X PATCH http://127.0.0.1:8000/v0/management/rotation \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <MANAGEMENT_API_KEY>' \
  -d '{"strategy":"round_robin"}'
```

Health states:

- `unchecked`: configured but not yet successfully checked.
- `healthy`: last check or request succeeded.
- `cooldown`: a transient upstream/request failure occurred; the token is skipped until cooldown expiry.
- `unhealthy`: DeepSeek rejected the token as invalid/expired.
- `disabled`: administratively disabled and never selected.

Routing rules:

- `fill_first` always chooses the first usable managed token in store order.
- `round_robin` advances across usable managed tokens.
- If a selected managed token fails before a completion stream is established due to authentication or an upstream 5xx protocol failure, the server marks that lease failed and retries once-through against another available managed token.
- If managed tokens exist, they take precedence over `DEEPSEEK_AUTH_TOKEN`; the static token remains a backward-compatible fallback only when the managed store is empty.
- The token store is designed for one server process. There is atomic replacement on write, but no cross-process file locking.

Open `http://127.0.0.1:8000/dashboard` for a simple browser UI to add, rename, replace, enable/disable, delete, check, and rotate tokens.

## Text request

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <API_KEY>' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say hello."}]}'
```

Omit the `Authorization` header only if `API_KEY` is not configured.

## Streaming request

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <API_KEY>' \
  -d '{"model":"deepseek-v4-flash","stream":true,"messages":[{"role":"user","content":"Say hello."}]}'
```

## Remote image request

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <API_KEY>' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image."},{"type":"image_url","image_url":{"url":"https://www.gstatic.com/webp/gallery/1.jpg"}}]}]}'
```

## Local image request

```bash
IMAGE_B64="$(base64 -w0 ./image.jpg)"
```

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <API_KEY>' \
  -d "{\"model\":\"deepseek-v4-flash\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$IMAGE_B64\"}}]}]}"
```

## Models and options

Primary aliases:

- `deepseek-v4-flash`
- `deepseek-v4-pro`

Aliases containing `think`, `r1`, or `reasoner` enable thinking. Aliases containing `search`, or requests with `web_search: true`, enable web search. Requests containing image parts use DeepSeek's vision transport automatically by uploading each image and sending `ref_file_ids` with `model_type: "vision"` upstream.

## Remote image safety

Remote images must use absolute HTTP(S) URLs. By default, the downloader:

- resolves hostnames before fetching;
- blocks loopback, private, link-local, reserved, and other non-global IP ranges;
- revalidates every redirect target;
- limits redirect count;
- requires an `image/*` MIME type;
- enforces `MAX_IMAGE_BYTES` while streaming.

Only set `ALLOW_PRIVATE_IMAGE_URLS=true` for trusted private deployments.

## Docker

```bash
docker build -t deepseek-python-api .
```

```bash
docker volume create deepseek-python-api-data
```

```bash
docker run --rm -p 8000:8000 \
  -e DEEPSEEK_AUTH_TOKEN='<deepseek-web-token>' \
  -e API_KEY='<client-api-key>' \
  -e MANAGEMENT_API_KEY='<management-api-key>' \
  -v deepseek-python-api-data:/data \
  deepseek-python-api
```

## Benchmark

The package includes a conservative end-to-end benchmark. It checks `/health` and `/v1/models` first, then measures the configured completion path. The benchmark never prints API keys or response bodies. Results include success rate, request rate, latency percentiles, and streaming time-to-first-event.

Run a small non-streaming smoke benchmark against a running local server:

```bash
.venv/bin/deepseek-python-api-benchmark --api-key '<API_KEY>'
```

Run a streaming benchmark:

```bash
.venv/bin/deepseek-python-api-benchmark --stream --requests 5 --concurrency 2 --api-key '<API_KEY>'
```

Save a machine-readable report:

```bash
.venv/bin/deepseek-python-api-benchmark --requests 10 --json benchmark.json --api-key '<API_KEY>'
```

Benchmark the optional request paths explicitly:

```bash
.venv/bin/deepseek-python-api-benchmark --model deepseek-v4-flash-think --reasoning-effort medium --api-key '<API_KEY>'
```

```bash
.venv/bin/deepseek-python-api-benchmark --model deepseek-v4-flash-search --web-search --api-key '<API_KEY>'
```

```bash
.venv/bin/deepseek-python-api-benchmark --image-url 'https://www.gstatic.com/webp/gallery/1.jpg' --api-key '<API_KEY>'
```

Use `--image-file ./image.jpg` for a local base64 image. Increase `--concurrency` only deliberately: this measures the entire proxy, proof-of-work, network, DeepSeek upstream, and token-routing path, not just local FastAPI overhead. Start with the defaults (`3` requests, one worker, one warmup), and avoid aggressive load against the unofficial upstream service.

The command exits nonzero if preflight fails, no request succeeds, or the success rate is below `--min-success-rate`. `DEEPSEEK_API_URL` and `API_KEY` may be supplied through the environment instead of command-line options.

## Verification commands

```bash
.venv/bin/ruff format .
```

```bash
.venv/bin/ruff check --fix .
```

```bash
.venv/bin/mypy src tests
```

```bash
.venv/bin/pytest tests --cov=deepseek_python_api --cov-report=term-missing
```

```bash
.venv/bin/python -m build
```

A resolved dependency snapshot can be generated with:

```bash
.venv/bin/python -m pip freeze --exclude-editable > requirements.lock
```

## License

GPL-3.0-or-later. The packaged proof-of-work WebAssembly asset comes from the GPL-licensed Chat2API project.
