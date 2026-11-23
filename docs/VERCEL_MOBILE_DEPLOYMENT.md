# Vercel & Mobile Remote Capture Deployment Guide (Lot B2, B3 & B4)

This guide explains how to deploy the mobile-first SENTRY capture UI statically on **Vercel** so you can snap or upload parking lot photos from your phone anywhere and send them directly to your local computer for storage and 4-model occupancy analysis across **69 active bays** (`B2`: 22, `B3`: 20, `B4`: 27).

---

## Architecture

```text
+-----------------------+        HTTPS (Vercel)        +-------------------------+
|  Phone / Field Browser| <--------------------------> |  Vercel Static Host     |
|  - Rear Camera Capture|                              |  (`ui/index.html` +     |
|  - Max 2048x1536 JPEG |                              |   `ui/samples/*`)       |
+-----------+-----------+                              +-------------------------+
            |
            | HTTPS POST (/api/upload or /api/analyze)
            | Headers: X-Sentry-Token + ngrok-skip-browser-warning
            v
+-----------------------+       HTTP (127.0.0.1)       +-------------------------+
|  HTTPS Tunnel         | ---------------------------> |  Local PC Python Server |
|  - cloudflared        |                              |  `run_lots_b2_b4_local` |
|  - localhost.run SSH  |                              |  - Saves to `uploads/`  |
|  - ngrok              |                              |  - Runs M1..M4 + CSV    |
+-----------------------+                              +-------------------------+
```

- **Same-origin Local Mode:** Opening `http://127.0.0.1:8080` directly connects with `apiBaseUrl = ""`.
- **Remote Tunnel Mode (Vercel -> Local PC):** The Vercel page stores your HTTPS tunnel URL (`sentry_api_url`) and optional access token (`sentry_api_token`) in `localStorage`, supports one-tap pairing links (`?api=<tunnel_url>&token=<token>`), and strips `?token=` from the browser address bar automatically after saving.

---

## 1. Deploy the UI to Vercel

The UI is zero-build static HTML/JS configured via [`vercel.json`](../vercel.json) (`outputDirectory: "ui"`).

### Option A — GitHub Import (Vercel Dashboard)
1. Push your repository to GitHub (`origin/main`).
2. In **Vercel -> Add New Project**, import your repository.
3. Keep **Root Directory** at `ai_services/parking_occupancy` (or if your Git repository root is already `ai_services/parking_occupancy`, leave it at `.`).
4. Leave **Build Command** empty (`vercel.json` already sets `framework: null` and `outputDirectory: "ui"`), then click **Deploy**.

### Option B — Vercel CLI
From `ai_services/parking_occupancy/`:

```bash
npx vercel --prod
```

---

## 2. Start the Local Computer Receiver

From `ai_services/parking_occupancy/` on the PC with your ONNX weights (`models/yolox_l.onnx`, `models/midas_small.onnx`):

```bash
python scripts/run_lots_b2_b4_local.py --serve --skip-initial-run --token "your-secret-token"
```

- `--serve`: Starts the HTTP receiver on `127.0.0.1:8080`.
- `--skip-initial-run`: Reuses `outputs/lots_b2_b4/sentry_b2_b4_report.json` if present so the server starts immediately.
- `--token`: Requires `X-Sentry-Token`, `Authorization: Bearer <token>`, or `?token=<token>` on `/api/*` (except `/api/health`) and `/outputs/*`. You can also set `SENTRY_UPLOAD_TOKEN`.
- Uploaded photos & metadata are saved under `outputs/lots_b2_b4/uploads/` as `<upload_id>.jpg` and `<upload_id>.meta.json`.

### Receiver Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Returns `{"ok": true, "mode": "lots_b2_b4_4model", "auth_required": ..., "authenticated": ..., "uploads_dir": "..."}` |
| `GET /api/uploads` | Lists the 30 newest uploaded images (`uploads/*.meta.json`) |
| `POST /api/upload` | Fast raw upload: saves photo + metadata on PC without running ONNX |
| `POST /api/analyze` | Saves fresh photo (or loads `{ upload_id }`), runs 4-model AI, updates CSV & 2D maps |

---

## 3. Expose Your Local PC Over HTTPS (Tunnel)

Run one of these in a second terminal while the Python server is running on port `8080`:

### Option 1 — Built-in Windows OpenSSH (`localhost.run`, zero install)
```bash
ssh -R 80:127.0.0.1:8080 nokey@localhost.run
```

### Option 2 — Cloudflare Quick Tunnel (`cloudflared`)
```bash
cloudflared tunnel --url http://127.0.0.1:8080
```

### Option 3 — ngrok
```bash
ngrok http 8080
```

Copy the `https://...` forwarding URL printed by your chosen tunnel tool.

---

## 4. Pair Your Phone & Capture Field Photos

1. Open your Vercel URL on desktop or phone and tap the top-bar **Computer Connection** status pill.
2. Paste your HTTPS tunnel URL (e.g. `https://xxxx.trycloudflare.com`) and your `--token`, then tap **Test & Save**.
3. Tap **Copy Mobile Pairing Link** (generates `https://<your-vercel-app>/?api=...&token=...`) and open/send that link to your phone — it auto-saves the connection and scrubs `?token=` from the URL bar.
4. *(Optional)* Use your phone browser's **Add to Home Screen** (`manifest.json` sets standalone app mode).
5. In the **Upload / Capture** tab on mobile:
   - Choose **Lot B2 (22 Bays)**, **Lot B3 (20 Bays)**, or **Lot B4 (27 Bays)**.
   - Tap **Take Photo** (opens rear camera) or **Choose Photo** (gallery). Large phone photos are automatically EXIF-oriented and resized client-side to max `2048x1536` JPEG before upload.
   - Tap **Upload Image to Computer** for quick raw transfer, or **Upload & Run 4-Model AI** to save the photo and run inference immediately (or analyze saved photos later from **Recent Uploads on Computer**).
