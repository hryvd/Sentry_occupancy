# Vercel & Mobile Remote Capture Deployment Guide (Lot B2, B3 & B4)

This guide explains how to deploy the mobile-first SENTRY capture UI statically on **Vercel**, connect it to your local PC receiver over an HTTPS tunnel, capture field photos anywhere from a phone, and store or analyze them across **69 active bays** (`B2`: 22, `B3`: 20, `B4`: 27).

---

## Architecture & How the Connection Works

```text
+-----------------------+        HTTPS (Vercel)        +-------------------------+
|  Phone / Field Browser| <--------------------------> |  Vercel Static Host     |
|  - Rear Camera Capture|                              |  (`ui/index.html` +     |
|  - Max 2048x1536 JPEG |                              |   `ui/samples/*`)       |
+-----------+-----------+                              +-------------------------+
            |
            | 1. OPTIONS preflight -> 204 CORS headers
            | 2. GET /api/health, GET /api/uploads
            | 3. POST /api/upload or POST /api/analyze (JSON + base64 JPEG)
            | 4. GET /outputs/... overlays (authenticated blob fetch)
            v
+-----------------------+       HTTP (127.0.0.1)       +-------------------------+
|  Cloudflare Tunnel    | ---------------------------> |  Local PC Python Server |
|  `cloudflared`        |     Loopback port 8080       |  `run_lots_b2_b4_local` |
|  (trycloudflare.com)  |                              |  - Saves to `uploads/`  |
|  Unthrottled QUIC/H2  |                              |  - Runs M1..M4 + CSV    |
+-----------------------+                              +-------------------------+
```

### End-to-End Request Lifecycle
1. **Static UI load on Vercel (`vercel.json` -> `ui/`):**
   - `vercel.json` serves `ui/index.html`, `ui/manifest.json`, and bundled sample images in `ui/samples/` with zero server-side routes on Vercel.
   - `ui/index.html` embeds the baseline 69-bay report (`EMBEDDED_REPORT`) so the map, slot inspector, and 14-column CSV preview work offline even before pairing with a PC.
2. **Tunnel URL & token bootstrap (`initConnectionFromParamsAndStorage()`):**
   - On load (`DOMContentLoaded`), the UI reads `?api=...&token=...` from the URL or falls back to `localStorage` (`sentry_api_url`, `sentry_api_token`).
   - If `?api=` or `?token=` is present in the URL, it normalizes and saves them into `localStorage` and scrubs `?token=` from the browser address bar with `window.history.replaceState`.
   - On a non-local hostname (such as `*.vercel.app`) when `sentry_api_url` is empty, the UI automatically expands `#connectionPanel` so you can paste your PC tunnel URL immediately.
3. **URL normalization & safety guards (`ui/index.html`):**
   - `normalizeApiBase(raw)` automatically prepends `https://` (`http://` for `localhost`/`127.0.0.1`) if the protocol scheme is omitted, and strips accidental `/api/...` or trailing slashes.
   - `syncConnectionInputsToState()` persists whatever you typed into `#apiUrlInput` and `#apiTokenInput` before any API call—so even if you forget to tap **Test & Save**, uploading still uses the typed tunnel URL.
   - `ensureRemoteTargetConfigured()` blocks uploads from going to Vercel itself (`/api/upload` -> 404) when no tunnel URL is configured, opens `#connectionPanel`, and prompts clearly.
   - `parseJsonResponse()` reads raw text first so plain-text HTTP/gateway errors (like Vercel 404 or tunnel 502) show the HTTP status, target origin, and snippet instead of `Unexpected token 'T'`.
4. **Client-side image optimization (`optimizeImageForUpload()`):**
   - Large mobile photos are decoded in browser canvas, oriented upright, resized to fit within `2048x1536`, and re-encoded as `image/jpeg` quality `0.88` before JSON base64 transmission.
5. **Cross-origin HTTP, Auth & Storage (`scripts/run_lots_b2_b4_local.py`):**
   - Binds strictly to loopback `127.0.0.1:8080` (`ThreadingHTTPServer`)—only the local machine or your active outbound tunnel can reach it.
   - `do_OPTIONS` responds `204` with `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods: GET, POST, OPTIONS`, and `Access-Control-Allow-Headers: Content-Type, Authorization, X-Sentry-Token, ngrok-skip-browser-warning`.
   - When `--token` is enabled, `/api/*` (except `/api/health`) and `/outputs/*` require matching `X-Sentry-Token`, `Authorization: Bearer <token>`, or `?token=<token>` verified via `hmac.compare_digest`.
   - `POST /api/upload` and `POST /api/analyze` enforce a `32 MB` body cap, write `<upload_id>.jpg` and `<upload_id>.meta.json` under `outputs/lots_b2_b4/uploads/`, and (`/api/analyze`) run M1..M4 + YOLO/MiDaS to update CSVs and overlay PNG/JPGs.
   - Remote overlay images (`/outputs/...`) are fetched by `loadAuthenticatedImage()` via `fetch()` with `X-Sentry-Token` and rendered through temporary `URL.createObjectURL(blob)` URLs (so `<img src>` works behind token auth).

- **Same-origin Local Mode:** Opening `http://127.0.0.1:8080` directly connects with `apiBaseUrl = ""` (same origin, no tunnel needed).

---

## 1. Deploy the UI to Vercel

The UI is zero-build static HTML/JS configured via [`vercel.json`](../vercel.json) (`outputDirectory: "ui"`).

### Option A — GitHub Import (Vercel Dashboard)
1. Push your repository to GitHub (`origin/main`).
2. In **Vercel -> Add New Project**, import your repository.
3. Keep **Root Directory** at `ai_services/parking_occupancy` (or `.` if your Git repository root is already `ai_services/parking_occupancy`).
4. Leave **Build Command** empty (`vercel.json` sets `framework: null` and `outputDirectory: "ui"`), then click **Deploy**.

### Option B — Vercel CLI
From `ai_services/parking_occupancy/`:

```bash
npx vercel --prod
```

---

## 2. Quick Start on Windows (One Command)

From `ai_services/parking_occupancy/` in PowerShell:

```powershell
.\scripts\start_pc_receiver.ps1 -Token "sentry-field-2026"
```

What [`scripts/start_pc_receiver.ps1`](../scripts/start_pc_receiver.ps1) does automatically:
1. Checks that port `8080` is free (exits cleanly if already in use).
2. Downloads standalone `cloudflared-windows-amd64.exe` into `%USERPROFILE%\.local\bin\cloudflared.exe` if missing (no admin/UAC required).
3. Starts `python scripts/run_lots_b2_b4_local.py --serve --skip-initial-run --host 127.0.0.1 --port 8080 --token "sentry-field-2026"`.
4. Starts `cloudflared tunnel --url http://127.0.0.1:8080 --no-autoupdate` and prints your live `https://<subdomain>.trycloudflare.com` URL and one-tap `?api=...&token=...` pairing suffix.
5. Stops both processes cleanly when you press `Ctrl+C`.

---

## 3. Manual Setup (2 Terminals)

### Terminal A — Start the Local Computer Receiver
From `ai_services/parking_occupancy/` on the PC with your ONNX weights (`models/yolox_l.onnx`, `models/midas_small.onnx`):

```bash
python scripts/run_lots_b2_b4_local.py --serve --skip-initial-run --host 127.0.0.1 --port 8080 --token "sentry-field-2026"
```

- `--serve`: Starts the HTTP receiver on `127.0.0.1:8080`.
- `--skip-initial-run`: Reuses `outputs/lots_b2_b4/sentry_b2_b4_report.json` if present so the server starts immediately.
- `--token`: Enables token protection (`SENTRY_UPLOAD_TOKEN` env var is also supported).

#### Receiver Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Returns `{"ok": true, "mode": "lots_b2_b4_4model", "auth_required": ..., "authenticated": ..., "uploads_dir": "..."}` |
| `GET /api/lots` | Returns the latest `sentry_b2_b4_report.json` payload |
| `GET /api/uploads` | Lists the 30 newest uploaded images (`uploads/*.meta.json`) |
| `GET /api/csv` | Downloads the combined 69-bay CSV report |
| `POST /api/upload` | Fast raw upload: saves photo + metadata on PC without running ONNX |
| `POST /api/analyze` | Saves fresh photo (or loads `{ upload_id }`), runs 4-model AI, updates CSV & 2D maps |

### Terminal B — Expose Over HTTPS

#### Option 1 — Cloudflare Quick Tunnel (`cloudflared`, Recommended)
Free, requires no Cloudflare account, and does **not** throttle photo uploads:

```powershell
# One-time user-local download (no admin rights needed)
New-Item -ItemType Directory -Force "$env:USERPROFILE\.local\bin" | Out-Null
Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "$env:USERPROFILE\.local\bin\cloudflared.exe"

# Start tunnel
& "$env:USERPROFILE\.local\bin\cloudflared.exe" tunnel --url http://127.0.0.1:8080 --no-autoupdate
```

Copy the `https://....trycloudflare.com` URL printed in the banner. Note that Quick Tunnel URLs rotate each time the tunnel restarts.

#### Option 2 — `localhost.run` SSH (Built-in Windows OpenSSH)
Zero install, but **free tier enforces bandwidth throttling** (`requests to your tunnel have been slowed...`) and may forcibly reset connections (`WinError 10054`) on large image uploads:

```bash
ssh -R 80:127.0.0.1:8080 nokey@localhost.run
```

#### Option 3 — `ngrok`
```bash
ngrok http 8080
```
(`ui/index.html` already sends `ngrok-skip-browser-warning: true`.)

---

## 4. Pair Your Phone & Capture Field Photos

1. Open your Vercel URL on desktop or phone. If no tunnel is configured yet, **Computer Connection** opens automatically.
2. Paste your HTTPS tunnel URL (e.g. `https://xxxx.trycloudflare.com`) and access token (`sentry-field-2026`), then tap **Test & Save**.
3. Tap **Copy Mobile Pairing Link** (generates `https://<your-vercel-app>/?api=...&token=...`) and open/send that link to your phone — it saves both values to `localStorage` and removes `?token=` from the URL bar.
4. *(Optional)* Use your phone browser's **Add to Home Screen** (`manifest.json` sets standalone app mode).
5. In the **Upload / Capture** tab on mobile:
   - Choose **Lot B2 (22 Bays)**, **Lot B3 (20 Bays)**, or **Lot B4 (27 Bays)**.
   - Tap **Take Photo** (opens rear camera) or **Choose Photo** (gallery).
   - Tap **Upload Image to Computer** for fast transfer, or **Upload & Run 4-Model AI** to save the photo and run inference immediately (or analyze saved photos later from **Recent Uploads on Computer**).

---

## 5. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Unexpected token 'T', "The page c"... is not valid JSON` | Request hit Vercel (`404 "The page could not be found"`) because tunnel URL was blank, unsaved, or missing `https://`. | Open **Computer Connection**, paste your full `https://...` tunnel URL (latest UI auto-adds `https://`), and tap **Test & Save**. |
| `Upload/AI failed: Failed to fetch` | Tunnel stopped/rotated, PC receiver isn't running on port `8080`, or `localhost.run` free tier throttled/dropped the connection (`WinError 10054`). | Switch to **Cloudflare Quick Tunnel (`cloudflared`)** via `.\scripts\start_pc_receiver.ps1`, copy the fresh `https://....trycloudflare.com` URL into Vercel, and re-test. |
| `Auth Required` / `HTTP 401` | Browser token doesn't match the PC receiver's `--token`. | Enter the exact `--token` value in **Computer Connection** and tap **Test & Save**. |
| Check / stop active PC receiver | Need to confirm whether port `8080` is occupied or stop a detached server. | Run `Get-NetTCPConnection -LocalPort 8080 -State Listen` in PowerShell to find the owning PID, or press `Ctrl+C` in the receiver/launcher terminal. |
