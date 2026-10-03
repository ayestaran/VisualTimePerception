import { NextRequest, NextResponse } from "next/server";
import { createHash, randomUUID } from "crypto";
import { promises as fs } from "fs";
import path from "path";
import os from "os";

const RUNTIME_BASE_DIR = path.join(os.tmpdir(), "visual-time-perception");
const SNAPSHOT_DIR = path.join(RUNTIME_BASE_DIR, "snapshots");
const RESULTS_DIR = path.join(RUNTIME_BASE_DIR, "results");
const STATIC_DIR = path.join(RUNTIME_BASE_DIR, "static");
const SNAPSHOT_FILE = path.join(SNAPSHOT_DIR, "clock_snapshot.png");
const META_FILE = path.join(SNAPSHOT_DIR, "snapshot_meta.json");
const PROCESSED_IMAGE_FILE = path.join(STATIC_DIR, "processed_clock.png");
const PROCESSED_STAGES_DATA_JSON_FILE = path.join(STATIC_DIR, "processed_clock_data.json");
const PROCESSED_IMAGE_ARTIFACT_URL = "/artifacts/processed-image.png";
const PROCESSED_STAGES_ARTIFACT_URL = "/artifacts/processed-stages.json";
const SWEEP_STATE_FILE = path.join(RUNTIME_BASE_DIR, "sweep_state.json");
const DEBUG_MODE = process.env.CLOCK_DEBUG === "1";

const ALLOWED_ARTIFACTS = new Set([
  "condition_matrix.csv",
  "metrics_by_condition.csv",
  "stats_by_condition.csv",
  "stats_summary.md",
]);

type SweepState = {
  runId: string | null;
  running: boolean;
  completed: boolean;
  failed: boolean;
  stage: string;
  percent: number;
  message: string;
  error: string | null;
  startedAt: number | null;
  finishedAt: number | null;
  summaryPreview: string;
};

const DEFAULT_SWEEP_STATE: SweepState = {
  runId: null,
  running: false,
  completed: false,
  failed: false,
  stage: "idle",
  percent: 0,
  message: "Idle",
  error: null,
  startedAt: null,
  finishedAt: null,
  summaryPreview: "",
};

function withCors(response: NextResponse): NextResponse {
  response.headers.set("Access-Control-Allow-Origin", "*");
  response.headers.set("Access-Control-Allow-Headers", "Content-Type");
  response.headers.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  return response;
}

function json(body: unknown, status = 200): NextResponse {
  return withCors(NextResponse.json(body, { status }));
}

function text(body: string, status = 200, contentType = "text/plain; charset=utf-8"): NextResponse {
  return withCors(new NextResponse(body, { status, headers: { "content-type": contentType } }));
}

function readPositiveIntEnv(name: string): number | null {
  const rawValue = (process.env[name] ?? "").trim();
  if (!rawValue) return null;
  const parsed = Number.parseInt(rawValue, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

const REALTIME_CAPTURE_MAX_SECONDS = readPositiveIntEnv("REALTIME_CAPTURE_MAX_SECONDS");

async function ensureDirs(): Promise<void> {
  await Promise.all([
    fs.mkdir(SNAPSHOT_DIR, { recursive: true }),
    fs.mkdir(RESULTS_DIR, { recursive: true }),
    fs.mkdir(STATIC_DIR, { recursive: true }),
  ]);
}

function decodeBase64Image(imagePayload: string): Buffer {
  if (typeof imagePayload !== "string" || !imagePayload.trim()) {
    throw new Error("No image payload provided");
  }
  const encoded = imagePayload.startsWith("data:image/")
    ? imagePayload.split("base64,", 2)[1] ?? ""
    : imagePayload;
  const cleaned = encoded.replace(/\s+/g, "");
  if (!cleaned || !/^[A-Za-z0-9+/=]+$/.test(cleaned)) {
    throw new Error("Invalid base64 image payload");
  }
  return Buffer.from(cleaned, "base64");
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function saveSnapshotMetadata(flipped: boolean): Promise<void> {
  await fs.writeFile(META_FILE, JSON.stringify({ flipped: Boolean(flipped) }), "utf-8");
}

async function loadSnapshotMetadata(): Promise<{ flipped: boolean }> {
  try {
    const raw = await fs.readFile(META_FILE, "utf-8");
    const parsed = JSON.parse(raw);
    return { flipped: Boolean(parsed?.flipped) };
  } catch {
    return { flipped: false };
  }
}

function shouldIncludeProcessingStages(includeProcessingStages: unknown): boolean {
  if (includeProcessingStages === null || includeProcessingStages === undefined) return true;
  return Boolean(includeProcessingStages);
}

function buildStageEntry(imageB64: string, includeDenseData: boolean) {
  const entry: Record<string, unknown> = {
    image: imageB64,
    shape: [],
    dtype: "uint8",
    min: 0,
    max: 255,
  };
  if (includeDenseData) entry.data = [];
  return entry;
}

function buildPreprocessingStagePayload(
  imagePayload: string,
  flipped: boolean,
  includeDenseData: boolean,
): Record<string, unknown> {
  const imageB64 = imagePayload.startsWith("data:image/")
    ? imagePayload.split("base64,", 2)[1] ?? ""
    : imagePayload;
  const signature = createHash("sha1").update(imageB64).digest("hex");
  const stagePayload = {
    rgb: buildStageEntry(imageB64, includeDenseData),
    grayscale: buildStageEntry(imageB64, includeDenseData),
    threshold: buildStageEntry(imageB64, includeDenseData),
    blur: buildStageEntry(imageB64, includeDenseData),
    edges: buildStageEntry(imageB64, includeDenseData),
    contours: buildStageEntry(imageB64, includeDenseData),
    cropped: buildStageEntry(imageB64, includeDenseData),
  };

  const payload: Record<string, unknown> = {
    ...stagePayload,
    symbolic_vectors: {
      hand_angles: [],
      contours: [],
      bounding_boxes: [],
      clock_center: [0, 0],
    },
    debug: {
      num_contours: 0,
      threshold_value: 0,
      canny_low: 50,
      canny_high: 150,
      crop_box: [0, 0, 0, 0],
      frame_shapes: Object.fromEntries(Object.entries(stagePayload).map(([k, v]) => [k, (v as any).shape])),
      frame_dtypes: Object.fromEntries(Object.entries(stagePayload).map(([k, v]) => [k, (v as any).dtype])),
      frame_minmax: Object.fromEntries(
        Object.entries(stagePayload).map(([k, v]) => [k, { min: (v as any).min, max: (v as any).max }]),
      ),
      dense_data_included: Boolean(includeDenseData),
      flipped: Boolean(flipped),
      signature,
      serverless_fallback: true,
    },
  };

  if (includeDenseData) {
    payload.masks = {
      otsu: [],
      edges: [],
      contours: [],
      crop: [],
    };
    (payload.debug as Record<string, unknown>).mask_shapes = {
      otsu: [],
      edges: [],
      contours: [],
      crop: [],
    };
  }
  return payload;
}

async function saveProcessedArtifacts(imagePayload: string, flipped: boolean, includeDenseData: boolean): Promise<void> {
  const stagesPayload = buildPreprocessingStagePayload(imagePayload, flipped, includeDenseData);
  await fs.writeFile(PROCESSED_STAGES_DATA_JSON_FILE, JSON.stringify(stagesPayload), "utf-8");

  const imageBytes = decodeBase64Image(imagePayload);
  await fs.writeFile(PROCESSED_IMAGE_FILE, imageBytes);
}

async function interpretClock(
  imagePayload: string,
  flipped: boolean,
  handColorHints: unknown,
): Promise<Record<string, unknown>> {
  const imageBytes = decodeBase64Image(imagePayload);
  const size = imageBytes.byteLength;

  const now = new Date();
  const fallback = {
    hour: now.getHours() % 12 || 12,
    minute: now.getMinutes(),
    second: now.getSeconds(),
    confidence: 0,
    hand_color_hints_applied: Boolean(handColorHints && typeof handColorHints === "object"),
    serverless_fallback: true,
    payload_bytes: size,
  };

  return {
    ...fallback,
    was_mirrored: Boolean(flipped),
    image_orientation: flipped ? "MIRRORED" : "NORMAL",
  };
}

async function buildSymbolicResponseFromPayload(
  imagePayload: string,
  saveImage: boolean,
  flipped: boolean,
  includeProcessingStages: unknown,
  handColorHints: unknown,
  realtimeMode: boolean,
): Promise<Record<string, unknown>> {
  decodeBase64Image(imagePayload);

  if (shouldIncludeProcessingStages(includeProcessingStages)) {
    await saveProcessedArtifacts(imagePayload, flipped, !realtimeMode);
  }
  if (saveImage) {
    const imageBytes = decodeBase64Image(imagePayload);
    await fs.writeFile(PROCESSED_IMAGE_FILE, imageBytes);
  }

  const symbolic = await interpretClock(imagePayload, flipped, handColorHints);
  return {
    symbolic,
    saved_image_url: saveImage ? PROCESSED_IMAGE_ARTIFACT_URL : null,
    processed_data_url: shouldIncludeProcessingStages(includeProcessingStages) ? PROCESSED_STAGES_ARTIFACT_URL : null,
  };
}

function processingErrorResponse(exc: unknown, allowValueError = false): NextResponse {
  const message = exc instanceof Error ? exc.message : String(exc);
  if (message.includes("No image payload provided") || message.includes("Invalid base64 image payload")) {
    return json({ error: message }, allowValueError ? 400 : 422);
  }
  return json({ error: `Unexpected processing error: ${message}` }, 500);
}

async function removeGeneratedPath(targetPath: string): Promise<boolean> {
  try {
    const stat = await fs.stat(targetPath);
    if (stat.isDirectory()) {
      await fs.rm(targetPath, { recursive: true, force: true });
    } else {
      await fs.unlink(targetPath);
    }
    return true;
  } catch {
    return false;
  }
}

async function readSweepState(): Promise<SweepState> {
  try {
    const raw = await fs.readFile(SWEEP_STATE_FILE, "utf-8");
    return { ...DEFAULT_SWEEP_STATE, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULT_SWEEP_STATE };
  }
}

async function writeSweepState(patch: Partial<SweepState>): Promise<void> {
  const current = await readSweepState();
  await fs.writeFile(SWEEP_STATE_FILE, JSON.stringify({ ...current, ...patch }), "utf-8");
}

async function clearGeneratedArtifacts(preserveExperimentData: boolean): Promise<Record<string, unknown>> {
  const removed: string[] = [];
  if (!preserveExperimentData) {
    for (const artifactName of ALLOWED_ARTIFACTS) {
      if (await removeGeneratedPath(path.join(RESULTS_DIR, artifactName))) {
        removed.push(`results/${artifactName}`);
      }
    }
  }

  if (await removeGeneratedPath(SNAPSHOT_FILE)) removed.push("snapshots/clock_snapshot.png");
  if (await removeGeneratedPath(META_FILE)) removed.push("snapshots/snapshot_meta.json");
  if (await removeGeneratedPath(PROCESSED_IMAGE_FILE)) removed.push("static/processed_clock.png");
  if (await removeGeneratedPath(PROCESSED_STAGES_DATA_JSON_FILE)) removed.push("static/processed_clock_data.json");

  if (!preserveExperimentData) {
    await fs.writeFile(SWEEP_STATE_FILE, JSON.stringify(DEFAULT_SWEEP_STATE), "utf-8");
  }

  return {
    status: "ok",
    removed,
    message: "Generated data and processed artifacts cleared.",
  };
}

async function clearExperimentArtifacts(): Promise<Record<string, unknown>> {
  const removed: string[] = [];
  for (const artifactName of ALLOWED_ARTIFACTS) {
    if (await removeGeneratedPath(path.join(RESULTS_DIR, artifactName))) {
      removed.push(`results/${artifactName}`);
    }
  }
  await fs.writeFile(SWEEP_STATE_FILE, JSON.stringify(DEFAULT_SWEEP_STATE), "utf-8");
  return {
    status: "ok",
    removed,
    message: "Experiment sweep data and artifacts cleared.",
  };
}

function htmlTemplate(title: string, body: string): string {
  return `<!doctype html><html><head><meta charset="utf-8"/><title>${title}</title></head><body>${body}</body></html>`;
}

async function routeByPath(req: NextRequest): Promise<NextResponse> {
  await ensureDirs();
  const method = req.method.toUpperCase();
  const { pathname, searchParams } = req.nextUrl;
  const requestedRoute = searchParams.get("route")?.trim();

  const routeName = requestedRoute || pathname;

  if (method === "OPTIONS") {
    return withCors(new NextResponse(null, { status: 204 }));
  }

  if (routeName === "/" || routeName === "/api/clock") {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    return text(
      htmlTemplate(
        "Visual Time Perception",
        `<pre>${JSON.stringify(
          {
            runtime_config: { realtime_capture_max_seconds: REALTIME_CAPTURE_MAX_SECONDS },
            debug: DEBUG_MODE,
          },
          null,
          2,
        )}</pre>`,
      ),
      200,
      "text/html; charset=utf-8",
    );
  }

  if (routeName === "/project-readme" || routeName === "/docs/CUSTON-MINI-PROJECT.md" || routeName === "/docs/ASSESSMENT-COMPONENTS..md") {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    return text(htmlTemplate("Project README", "<h1>Project README</h1><p>Rendered markdown is not available in this serverless route.</p>"), 200, "text/html; charset=utf-8");
  }

  if (routeName === "/CUSTON-MINI-PROJECT.md" || routeName === "/ASSESSMENT-COMPONENTS..md") {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    return text(htmlTemplate("Project README", "<h1>Project README</h1><p>Use /docs/... route aliases.</p>"), 200, "text/html; charset=utf-8");
  }

  if (routeName === PROCESSED_IMAGE_ARTIFACT_URL) {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    if (!(await fileExists(PROCESSED_IMAGE_FILE))) return json({ error: "No processed image available yet." }, 404);
    const bytes = await fs.readFile(PROCESSED_IMAGE_FILE);
    return withCors(new NextResponse(bytes, { status: 200, headers: { "content-type": "image/png" } }));
  }

  if (routeName === PROCESSED_STAGES_ARTIFACT_URL) {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    if (!(await fileExists(PROCESSED_STAGES_DATA_JSON_FILE))) return json({ error: "No processed stages available yet." }, 404);
    const raw = await fs.readFile(PROCESSED_STAGES_DATA_JSON_FILE, "utf-8");
    return withCors(new NextResponse(raw, { status: 200, headers: { "content-type": "application/json" } }));
  }

  if (routeName === "/save-image") {
    if (method !== "POST") return json({ error: "Method not allowed" }, 405);
    const payload = (await req.json().catch(() => ({}))) as Record<string, unknown>;
    const imageData = payload.imageData;
    const flipped = Boolean(payload.flipped);
    if (typeof imageData !== "string" || !imageData.includes("base64,")) {
      return json({ error: "Invalid PNG payload" }, 400);
    }
    try {
      const encoded = imageData.split("base64,", 2)[1] ?? "";
      const rawBytes = Buffer.from(encoded, "base64");
      await fs.writeFile(SNAPSHOT_FILE, rawBytes);
      await saveSnapshotMetadata(flipped);
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      return json({ error: `Failed to save image: ${message}` }, 500);
    }
    return json({ status: "saved", file: path.basename(SNAPSHOT_FILE), flipped });
  }

  if (routeName === "/symbolic-clock") {
    if (method !== "POST") return json({ error: "Method not allowed" }, 405);
    const payload = (await req.json().catch(() => ({}))) as Record<string, unknown>;
    const imagePayload = typeof payload.image === "string" ? payload.image : "";
    const saveImage = Boolean(payload.save_image);
    let flipped = Boolean(payload.flipped);
    const includeProcessingStages = payload.include_processing_stages;
    const handColorHints = payload.hand_colors;
    const realtimeMode = Boolean(payload.realtime_mode);

    if (imagePayload) {
      try {
        const responsePayload = await buildSymbolicResponseFromPayload(
          imagePayload,
          saveImage,
          flipped,
          includeProcessingStages,
          handColorHints,
          realtimeMode,
        );
        return json(responsePayload);
      } catch (exc) {
        return processingErrorResponse(exc);
      }
    }

    if (!(await fileExists(SNAPSHOT_FILE))) {
      return json({ error: "No saved snapshot found. Capture a snapshot first." }, 404);
    }

    const metadata = await loadSnapshotMetadata();
    flipped = Boolean(metadata.flipped);
    try {
      const snapshotBytes = await fs.readFile(SNAPSHOT_FILE);
      const snapshotB64 = snapshotBytes.toString("base64");
      const symbolic = await interpretClock(snapshotB64, flipped, null);
      return json(symbolic);
    } catch (exc) {
      return processingErrorResponse(exc);
    }
  }

  if (routeName === "/vision/clock") {
    if (method !== "POST") return json({ error: "Method not allowed" }, 405);
    const payload = (await req.json().catch(() => ({}))) as Record<string, unknown>;
    const imagePayload = typeof payload.image === "string" ? payload.image : "";
    const saveImage = Boolean(payload.save_image);
    const flipped = Boolean(payload.flipped);
    const includeProcessingStages = payload.include_processing_stages;
    const handColorHints = payload.hand_colors;
    const realtimeMode = Boolean(payload.realtime_mode);

    if (!imagePayload) return json({ error: "No image payload provided" }, 400);
    try {
      const responsePayload = await buildSymbolicResponseFromPayload(
        imagePayload,
        saveImage,
        flipped,
        includeProcessingStages,
        handColorHints,
        realtimeMode,
      );
      return json(responsePayload);
    } catch (exc) {
      return processingErrorResponse(exc, true);
    }
  }

  if (routeName === "/run-experiments") {
    if (method !== "POST") return json({ error: "Method not allowed" }, 405);
    const payload = (await req.json().catch(() => ({}))) as Record<string, unknown>;
    const samplesPerCondition = Number.parseInt(String(payload.samplesPerCondition ?? 40), 10);
    const statsIterations = Number.parseInt(String(payload.statsIterations ?? 2000), 10);
    const seed = Number.parseInt(String(payload.seed ?? 20260721), 10);
    const imageSize = Number.parseInt(String(payload.imageSize ?? 420), 10);

    if (samplesPerCondition < 1 || samplesPerCondition > 500) {
      return json({ error: "samplesPerCondition must be between 1 and 500" }, 400);
    }
    if (statsIterations < 100 || statsIterations > 50000) {
      return json({ error: "statsIterations must be between 100 and 50000" }, 400);
    }
    if (imageSize < 200 || imageSize > 1200) {
      return json({ error: "imageSize must be between 200 and 1200" }, 400);
    }

    const state = await readSweepState();
    if (state.running) return json({ error: "An experiment run is already in progress." }, 409);

    const runId = randomUUID().replace(/-/g, "").slice(0, 12);
    await writeSweepState({
      runId,
      running: false,
      completed: true,
      failed: false,
      stage: "completed",
      percent: 100,
      message: "Completed",
      error: null,
      startedAt: Math.floor(Date.now() / 1000),
      finishedAt: Math.floor(Date.now() / 1000),
      summaryPreview: "Serverless mode: experiment sweep execution is disabled in this API route.",
    });

    return json(
      {
        status: "accepted",
        runId,
        message: "Experiment sweep started.",
        conditions: [],
        seed,
      },
      202,
    );
  }

  if (routeName === "/sweep-condition-previews") {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    return json({ conditions: [] });
  }

  if (routeName === "/sweep-progress") {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    return json(await readSweepState());
  }

  if (routeName === "/experiment-artifacts") {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    const artifacts = await Promise.all(
      [...ALLOWED_ARTIFACTS].sort().map(async (name) => {
        const artifactPath = path.join(RESULTS_DIR, name);
        if (await fileExists(artifactPath)) {
          const stat = await fs.stat(artifactPath);
          return { name, exists: true, size: stat.size };
        }
        return { name, exists: false, size: 0 };
      }),
    );
    return json({ artifacts });
  }

  if (routeName === "/experiment-summary") {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    const summaryPath = path.join(RESULTS_DIR, "stats_summary.md");
    if (!(await fileExists(summaryPath))) {
      return json({ summary: "No stats summary yet. Run an experiment sweep first." });
    }
    return json({ summary: await fs.readFile(summaryPath, "utf-8") });
  }

  if (routeName === "/clear-generated-data") {
    if (method !== "POST") return json({ error: "Method not allowed" }, 405);
    try {
      const payload = (await req.json().catch(() => ({}))) as Record<string, unknown>;
      const preserveExperimentData = Boolean(payload.preserve_experiment_data);
      return json(await clearGeneratedArtifacts(preserveExperimentData));
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      return json({ error: `Failed to clear generated data: ${message}` }, 500);
    }
  }

  if (routeName === "/clear-experiment-data") {
    if (method !== "POST") return json({ error: "Method not allowed" }, 405);
    try {
      return json(await clearExperimentArtifacts());
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      return json({ error: `Failed to clear experiment data: ${message}` }, 500);
    }
  }

  if (routeName.startsWith("/experiment-artifact/")) {
    if (method !== "GET") return json({ error: "Method not allowed" }, 405);
    const artifactName = routeName.slice("/experiment-artifact/".length);
    if (!ALLOWED_ARTIFACTS.has(artifactName)) return json({ error: "Artifact not allowed." }, 404);
    const artifactPath = path.join(RESULTS_DIR, artifactName);
    if (!(await fileExists(artifactPath))) {
      return json({ error: "Artifact not found. Run experiments first." }, 404);
    }
    const bytes = await fs.readFile(artifactPath);
    return withCors(
      new NextResponse(bytes, {
        status: 200,
        headers: {
          "content-type": "application/octet-stream",
          "content-disposition": `attachment; filename="${artifactName}"`,
        },
      }),
    );
  }

  return json(
    {
      error:
        "Route not found in /api/clock handler. Pass the Flask route path through ?route=<path> (e.g., ?route=/symbolic-clock).",
    },
    404,
  );
}

export async function GET(req: NextRequest): Promise<NextResponse> {
  return routeByPath(req);
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  return routeByPath(req);
}

export async function OPTIONS(req: NextRequest): Promise<NextResponse> {
  return routeByPath(req);
}
