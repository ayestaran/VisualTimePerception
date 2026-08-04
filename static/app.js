const canvas = document.getElementById("clockCanvas");
const ctx = canvas ? canvas.getContext("2d") : null;
const flipBtn = document.getElementById("flipBtn");
const resetClockBtn = document.getElementById("resetClockBtn");
const randomThemeBtn = document.getElementById("randomThemeBtn");
const themeSelect = document.getElementById("themeSelect");
const symbolicBtn = document.getElementById("symbolicBtn");
const realtimeBtn = document.getElementById("realtimeBtn");
const exportLogBtn = document.getElementById("exportLogBtn");
const openReadmeBtn = document.getElementById("openReadmeBtn");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const randomizeEverySecondToggle = document.getElementById("randomizeEverySecondToggle");
const rotationIncrementToggle = document.getElementById("rotationIncrementToggle");
const randomFlipToggle = document.getElementById("randomFlipToggle");
const rotate3DToggle = document.getElementById("rotate3DToggle");
const extraLineToggle = document.getElementById("extraLineToggle");
const artifactSafeModeToggle = document.getElementById("artifactSafeModeToggle");
const groundingPresetLibrary = document.getElementById("groundingPresetLibrary");
const groundingScriptDropzone = document.getElementById("groundingScriptDropzone");
const groundingScriptSequenceEl = document.getElementById("groundingScriptSequence");
const groundingScriptOutput = document.getElementById("groundingScriptOutput");
const addAllGroundingPresetsBtn = document.getElementById("addAllGroundingPresetsBtn");
const applyGroundingScriptBtn = document.getElementById("applyGroundingScriptBtn");
const copyGroundingScriptBtn = document.getElementById("copyGroundingScriptBtn");
const clearGroundingScriptBtn = document.getElementById("clearGroundingScriptBtn");
const histogramCanvas = document.getElementById("histogramCanvas");
const histogramCtx = histogramCanvas ? histogramCanvas.getContext("2d") : null;
const symbolicOutput = document.getElementById("symbolicOutput");
const probabilisticReasoningOutput = document.getElementById("probabilisticReasoningOutput");
const symbolicLogOutput = document.getElementById("symbolicLogOutput");
const processedImagePreview = document.getElementById("processedImagePreview");
const processedImageDownloadLink = document.getElementById("processedImageDownloadLink");
const opencvStagesDownloadLink = document.getElementById("opencvStagesDownloadLink");
const opencvStageSelect = document.getElementById("opencvStageSelect");
const opencvStagesList = document.getElementById("opencvStagesList");
const clearDataBtn = document.getElementById("clearDataBtn");
const clearWorkspaceBtn = document.getElementById("clearWorkspaceBtn");
const expSamplesInput = document.getElementById("expSamplesInput");
const expStatsIterationsInput = document.getElementById("expStatsIterationsInput");
const runExperimentsBtn = document.getElementById("runExperimentsBtn");
const clearExperimentsBtn = document.getElementById("clearExperimentsBtn");
const refreshExperimentsBtn = document.getElementById("refreshExperimentsBtn");
const downloadConditionMatrixBtn = document.getElementById("downloadConditionMatrixBtn");
const downloadMetricsBtn = document.getElementById("downloadMetricsBtn");
const downloadStatsCsvBtn = document.getElementById("downloadStatsCsvBtn");
const downloadStatsSummaryBtn = document.getElementById("downloadStatsSummaryBtn");
const experimentsStatus = document.getElementById("experimentsStatus");
const experimentsSummaryOutput = document.getElementById("experimentsSummaryOutput");
const sweepProgressWrap = document.getElementById("sweepProgressWrap");
const sweepProgressTrack = document.getElementById("sweepProgressTrack");
const sweepProgressFill = document.getElementById("sweepProgressFill");
const sweepProgressLabel = document.getElementById("sweepProgressLabel");
const sweepStageList = document.getElementById("sweepStageList");
const sweepConditionPreviews = document.getElementById("sweepConditionPreviews");
const realtimeLimitInfo = document.getElementById("realtimeLimitInfo");

const runtimeConfig = (typeof window !== "undefined" && window.APP_RUNTIME_CONFIG) || {};
const configuredRealtimeMaxSecondsRaw = Number.parseInt(runtimeConfig.realtimeCaptureMaxSeconds, 10);
const REALTIME_MAX_SECONDS = Number.isFinite(configuredRealtimeMaxSecondsRaw) && configuredRealtimeMaxSecondsRaw > 0
  ? configuredRealtimeMaxSecondsRaw
  : null;
const passwordPromptRequirements = runtimeConfig && typeof runtimeConfig.passwordPromptRequirements === "object"
  ? runtimeConfig.passwordPromptRequirements
  : {};

let isFlipped = false;
let lastCaptureAt = null;
let realtimeTimer = null;
let realtimeBusy = false;
let realtimeFrameCount = 0;
let realtimeRunStartedAt = null;
let realtimeMaxStopTimer = null;
let processedPreviewUpdateSeq = 0;
let openCvStagesUpdateSeq = 0;
let experimentsBusy = false;
let sweepPollTimer = null;
let sweepProgressValue = 0;
const REALTIME_INTERVAL_MS = 1000;
const REALTIME_PREVIEW_REFRESH_EVERY = 1;
const REALTIME_STAGE_REFRESH_EVERY = 1;
const symbolicHistory = [];
const processingHistory = [];
const MAX_SYMBOLIC_HISTORY = 80;
const MAX_PROCESSING_HISTORY = 80;
let histogramDrawRafId = null;
let histogramDrawInProgress = false;
const THEME_STORAGE_KEY = "clock-theme";
const RANDOMIZE_EVERY_SECOND_STORAGE_KEY = "clock-randomize-every-second";
const ROTATION_INCREMENT_STORAGE_KEY = "clock-rotation-increment";
const RANDOM_FLIP_STORAGE_KEY = "clock-random-flip";
const ROTATE_3D_STORAGE_KEY = "clock-rotate-3d";
const EXTRA_LINE_STORAGE_KEY = "clock-extra-line";
const ARTIFACT_SAFE_MODE_STORAGE_KEY = "clock-artifact-safe-mode";
const CLOCK_FACE_STATE_STORAGE_KEY = "clock-face-state";
const GROUNDING_SCRIPT_SEQUENCE_STORAGE_KEY = "clock-grounding-script-sequence";
const SYMBOLIC_HISTORY_STORAGE_KEY = "clock-symbolic-history";
const SWEEP_PREVIEWS_STORAGE_KEY = "clock-sweep-condition-previews";
const PROCESSING_HISTORY_STORAGE_KEY = "clock-processing-history";
const SYMBOLIC_OUTPUT_STORAGE_KEY = "clock-symbolic-output";
const PROBABILISTIC_OUTPUT_STORAGE_KEY = "clock-probabilistic-output";
const ROTATION_INCREMENT_DEGREES = 6;
const ROTATE_3D_INCREMENT_DEGREES = 12;
const NON_FATAL_HAND_DETECTION_ERROR_TEXT = "Unable to detect enough clock hands in the snapshot";
const OPENCV_STAGE_ORDER = ["rgb", "grayscale", "threshold", "blur", "edges", "contours", "cropped"];
const OPENCV_STAGE_LABELS = {
  rgb: "RGB",
  grayscale: "Grayscale",
  threshold: "Threshold",
  blur: "Blur",
  edges: "Edges",
  contours: "Contours",
  cropped: "Cropped",
};
const DEFAULT_OPENCV_STAGE_KEY = "edges";
const GROUNDING_PRESETS = [
  {
    id: "baseline-clean-grounding",
    title: "Baseline Clean Grounding",
    summary: "Stabilize orientation and remove visual distractors for anchor-quality data.",
    settings: {
      isFlipped: false,
      clockRotationDegrees: 0,
      clock3DRotationDegrees: 0,
      randomizeThemeEverySecond: false,
      rotateClockEverySecond: false,
      randomFlipEverySecond: false,
      rotateClock3DEverySecond: false,
      showExtraDistractorLine: false,
      artifactSafeMode: true,
    },
  },
  {
    id: "roman-numeral-anchoring",
    title: "Roman Numeral Anchoring",
    summary: "Prioritize clean numeral grounding conditions for Roman-index themes.",
    settings: {
      randomizeThemeEverySecond: false,
      rotateClockEverySecond: false,
      randomFlipEverySecond: false,
      rotateClock3DEverySecond: false,
      showExtraDistractorLine: false,
      artifactSafeMode: true,
    },
  },
  {
    id: "theme-domain-shift",
    title: "Theme Domain Shift",
    summary: "Increase appearance diversity while preserving geometric stability.",
    settings: {
      randomizeThemeEverySecond: true,
      rotateClockEverySecond: false,
      randomFlipEverySecond: false,
      rotateClock3DEverySecond: false,
      artifactSafeMode: true,
    },
  },
  {
    id: "rotation-invariance",
    title: "Rotation Invariance",
    summary: "Train grounding across planar rotation changes.",
    settings: {
      rotateClockEverySecond: true,
      randomFlipEverySecond: false,
      rotateClock3DEverySecond: false,
      artifactSafeMode: true,
    },
  },
  {
    id: "flip-invariance",
    title: "Flip Invariance",
    summary: "Stress mirror-orientation grounding.",
    settings: {
      randomFlipEverySecond: true,
      rotateClockEverySecond: false,
      rotateClock3DEverySecond: false,
      artifactSafeMode: true,
    },
  },
  {
    id: "backview-invariance",
    title: "Back View Invariance",
    summary: "Force 3D back-view transitions to test orientation grounding robustness.",
    settings: {
      rotateClock3DEverySecond: true,
      artifactSafeMode: true,
    },
  },
  {
    id: "artifact-resilience-challenge",
    title: "Artifact Resilience Challenge",
    summary: "Introduce controlled artifacts and distractions for robustness training.",
    settings: {
      showExtraDistractorLine: true,
      artifactSafeMode: false,
      randomizeThemeEverySecond: true,
    },
  },
  {
    id: "plain-reference-reset",
    title: "Plain Reference Reset",
    summary: "Reset to plain baseline before a new grounding experiment.",
    settings: {
      resetClockFace: true,
    },
  },
];

let processedPreviewImgEl = null;
let opencvStagesInitialized = false;
const opencvStageElements = new Map();
let selectedOpenCvStageKey = DEFAULT_OPENCV_STAGE_KEY;
let latestOpenCvStagesPayload = null;

function getRealtimeLimitLabel() {
  if (REALTIME_MAX_SECONDS === null) {
    return "Realtime max session: unlimited";
  }

  if (realtimeTimer !== null && Number.isFinite(realtimeRunStartedAt)) {
    const elapsedSeconds = Math.max(0, Math.floor((Date.now() - realtimeRunStartedAt) / 1000));
    const remainingSeconds = Math.max(0, REALTIME_MAX_SECONDS - elapsedSeconds);
    return `Realtime max session: ${REALTIME_MAX_SECONDS}s (remaining ${remainingSeconds}s)`;
  }

  return `Realtime max session: ${REALTIME_MAX_SECONDS}s`;
}

function isPasswordPromptRequired(promptKey) {
  return false;
}

function isPasswordPromptRequiredWithFallback(primaryKey, fallbackKey) {
  if (Object.prototype.hasOwnProperty.call(passwordPromptRequirements, primaryKey)) {
    return passwordPromptRequirements[primaryKey] !== false;
  }
  return isPasswordPromptRequired(fallbackKey);
}

function updateRealtimeLimitInfo() {
  if (!realtimeLimitInfo) {
    return;
  }
  realtimeLimitInfo.textContent = getRealtimeLimitLabel();
}
const DEFAULT_PLAIN_CLOCK_THEME = {
  name: "Plain Classic",
  era: "classic",
  numberStyle: "classic",
  facePattern: "none",
  numberColor: "#10233a",
  numberFont: "600 28px Georgia, serif",
  hourHandStyle: "classic",
  minuteHandStyle: "classic",
  secondHandStyle: "classic",
  hourColor: "#0e2940",
  minuteColor: "#123e6b",
  secondColor: "#c32828",
  centerOuterColor: "#0e2940",
  centerInnerColor: "#c32828",
  tickColor: "#28435f",
  faceInnerColor: "#ffffff",
  faceOuterColor: "#d9ebff",
  bezelColor: "#1f5f8b",
};
const ERA_THEME_TEMPLATES = [
  {
    baseName: "Art Deco",
    era: "1930s",
    numberStyle: "romanDeco",
    numberFont: "600 28px Georgia, serif",
    facePattern: "decoRings",
    hourHandStyle: "spade1930",
    minuteHandStyle: "spade1930",
    secondHandStyle: "rail1930",
  },
  {
    baseName: "Mid-Century",
    era: "1950s",
    numberStyle: "midCentury",
    numberFont: "700 27px Garamond, serif",
    facePattern: "sunburst1950",
    hourHandStyle: "dauphine1950",
    minuteHandStyle: "dauphine1950",
    secondHandStyle: "needle1950",
  },
  {
    baseName: "Cushion Dial",
    era: "1970s",
    numberStyle: "retro1970",
    numberFont: "700 27px 'Times New Roman', Times, serif",
    facePattern: "target1970",
    hourHandStyle: "block1970",
    minuteHandStyle: "block1970",
    secondHandStyle: "needle1950",
  },
  {
    baseName: "Quartz Sport",
    era: "1980s",
    numberStyle: "quartz1980",
    numberFont: "700 26px Palatino, 'Palatino Linotype', serif",
    facePattern: "grid1980",
    hourHandStyle: "baton1980",
    minuteHandStyle: "baton1980",
    secondHandStyle: "lollipop1980",
  },
  {
    baseName: "Technical",
    era: "2000s",
    numberStyle: "tech2000",
    numberFont: "700 27px 'Trebuchet MS', 'Gill Sans', sans-serif",
    facePattern: "tech2000",
    hourHandStyle: "skeleton2000",
    minuteHandStyle: "skeleton2000",
    secondHandStyle: "lollipop1980",
  },
];

const THEME_VARIANTS = [
  {
    variantName: "Aegean",
    numberColor: "#18324b",
    hourColor: "#163149",
    minuteColor: "#2b5479",
    secondColor: "#c43a2c",
    centerOuterColor: "#163149",
    centerInnerColor: "#c43a2c",
    tickColor: "#355a7b",
    faceInnerColor: "#f8fbff",
    faceOuterColor: "#d7e8ff",
    bezelColor: "#2c5d8e",
  },
  {
    variantName: "Sage",
    numberColor: "#27443a",
    hourColor: "#1f3a31",
    minuteColor: "#35614e",
    secondColor: "#c15a2d",
    centerOuterColor: "#1f3a31",
    centerInnerColor: "#c15a2d",
    tickColor: "#4b7465",
    faceInnerColor: "#fbfdf8",
    faceOuterColor: "#dceddf",
    bezelColor: "#537c6a",
  },
  {
    variantName: "Copper",
    numberColor: "#4c3224",
    hourColor: "#472d22",
    minuteColor: "#6e4632",
    secondColor: "#2d78b8",
    centerOuterColor: "#472d22",
    centerInnerColor: "#2d78b8",
    tickColor: "#7f5a45",
    faceInnerColor: "#fff8f2",
    faceOuterColor: "#efd5c0",
    bezelColor: "#94674d",
  },
  {
    variantName: "Graphite",
    numberColor: "#303642",
    hourColor: "#2e3440",
    minuteColor: "#4e5668",
    secondColor: "#d98b1d",
    centerOuterColor: "#2e3440",
    centerInnerColor: "#d98b1d",
    tickColor: "#646f84",
    faceInnerColor: "#f4f7fb",
    faceOuterColor: "#d6dfed",
    bezelColor: "#5a687f",
  },
  {
    variantName: "Burgundy",
    numberColor: "#4a2232",
    hourColor: "#4a2232",
    minuteColor: "#6f3950",
    secondColor: "#2e7c98",
    centerOuterColor: "#4a2232",
    centerInnerColor: "#2e7c98",
    tickColor: "#7e5568",
    faceInnerColor: "#fff7fb",
    faceOuterColor: "#efd5e2",
    bezelColor: "#7a4a60",
  },
  {
    variantName: "Indigo",
    numberColor: "#22305a",
    hourColor: "#22305a",
    minuteColor: "#3a4f8f",
    secondColor: "#8db233",
    centerOuterColor: "#22305a",
    centerInnerColor: "#8db233",
    tickColor: "#5a69a3",
    faceInnerColor: "#f7f9ff",
    faceOuterColor: "#d8def4",
    bezelColor: "#4a5f9c",
  },
  {
    variantName: "Teal",
    numberColor: "#1f4a52",
    hourColor: "#1c434b",
    minuteColor: "#2e6570",
    secondColor: "#d14f3d",
    centerOuterColor: "#1c434b",
    centerInnerColor: "#d14f3d",
    tickColor: "#4f7f86",
    faceInnerColor: "#f5fcfd",
    faceOuterColor: "#d1eaed",
    bezelColor: "#4f8189",
  },
  {
    variantName: "Olive",
    numberColor: "#3c4723",
    hourColor: "#394321",
    minuteColor: "#5a6734",
    secondColor: "#b85d2b",
    centerOuterColor: "#394321",
    centerInnerColor: "#b85d2b",
    tickColor: "#6f7c47",
    faceInnerColor: "#fbfcf4",
    faceOuterColor: "#e4eac6",
    bezelColor: "#737f4d",
  },
  {
    variantName: "Ruby",
    numberColor: "#5a1e2d",
    hourColor: "#551c2a",
    minuteColor: "#7e2c41",
    secondColor: "#2a78a8",
    centerOuterColor: "#551c2a",
    centerInnerColor: "#2a78a8",
    tickColor: "#8a5064",
    faceInnerColor: "#fff6f8",
    faceOuterColor: "#f0d0d8",
    bezelColor: "#8b4459",
  },
  {
    variantName: "Sand",
    numberColor: "#5c4732",
    hourColor: "#57412f",
    minuteColor: "#7b6042",
    secondColor: "#2f74a8",
    centerOuterColor: "#57412f",
    centerInnerColor: "#2f74a8",
    tickColor: "#8e7458",
    faceInnerColor: "#fffaf4",
    faceOuterColor: "#eed9bc",
    bezelColor: "#9a7c5e",
  },
  {
    variantName: "Slate",
    numberColor: "#2a3547",
    hourColor: "#273243",
    minuteColor: "#3f526f",
    secondColor: "#d0662d",
    centerOuterColor: "#273243",
    centerInnerColor: "#d0662d",
    tickColor: "#5f7089",
    faceInnerColor: "#f5f8fc",
    faceOuterColor: "#d4deea",
    bezelColor: "#5a6d86",
  },
  {
    variantName: "Neon",
    numberColor: "#1d2f5f",
    hourColor: "#1b2d5a",
    minuteColor: "#2f4e99",
    secondColor: "#7dbf3a",
    centerOuterColor: "#1b2d5a",
    centerInnerColor: "#7dbf3a",
    tickColor: "#5a73b0",
    faceInnerColor: "#f4f8ff",
    faceOuterColor: "#d1ddf9",
    bezelColor: "#4a67ae",
  },
];

function buildClockThemes() {
  const themes = [];
  for (const eraTemplate of ERA_THEME_TEMPLATES) {
    for (const variant of THEME_VARIANTS) {
      themes.push({
        name: `${eraTemplate.era} ${eraTemplate.baseName} ${variant.variantName}`,
        era: eraTemplate.era,
        numberStyle: eraTemplate.numberStyle,
        numberFont: eraTemplate.numberFont,
        facePattern: eraTemplate.facePattern,
        hourHandStyle: eraTemplate.hourHandStyle,
        minuteHandStyle: eraTemplate.minuteHandStyle,
        secondHandStyle: eraTemplate.secondHandStyle,
        numberColor: variant.numberColor,
        hourColor: variant.hourColor,
        minuteColor: variant.minuteColor,
        secondColor: variant.secondColor,
        centerOuterColor: variant.centerOuterColor,
        centerInnerColor: variant.centerInnerColor,
        tickColor: variant.tickColor,
        faceInnerColor: variant.faceInnerColor,
        faceOuterColor: variant.faceOuterColor,
        bezelColor: variant.bezelColor,
      });
    }
  }

  return themes;
}

const CLOCK_THEMES = buildClockThemes();

let currentClockTheme = CLOCK_THEMES[0];
let randomizeThemeEverySecond = false;
let rotateClockEverySecond = false;
let clockRotationDegrees = 0;
let randomFlipEverySecond = false;
let rotateClock3DEverySecond = false;
let clock3DRotationDegrees = 0;
let showExtraDistractorLine = false;
let artifactSafeMode = true;
let groundingScriptSequence = [];
let lastKnownDebugMetrics = null;

const groundingPresetById = new Map(GROUNDING_PRESETS.map((preset) => [preset.id, preset]));

function applyTheme(theme) {
  if (theme === "light") {
    document.body.classList.add("light-theme");
  } else {
    document.body.classList.remove("light-theme");
  }

  if (themeToggleBtn) {
    themeToggleBtn.title = theme === "light" ? "Switch to dark theme" : "Switch to light theme";
  }

  drawProcessingHistogram();
}

function initializeTheme() {
  const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  if (savedTheme === "light" || savedTheme === "dark") {
    applyTheme(savedTheme);
    return;
  }

  localStorage.setItem(THEME_STORAGE_KEY, "dark");
  applyTheme("dark");
}

function initializeRandomizeEverySecondSetting() {
  const savedValue = localStorage.getItem(RANDOMIZE_EVERY_SECOND_STORAGE_KEY);
  randomizeThemeEverySecond = savedValue === "true";

  if (randomizeEverySecondToggle) {
    randomizeEverySecondToggle.checked = randomizeThemeEverySecond;
  }
}

function initializeRotationIncrementSetting() {
  const savedValue = localStorage.getItem(ROTATION_INCREMENT_STORAGE_KEY);
  rotateClockEverySecond = savedValue === "true";

  if (rotationIncrementToggle) {
    rotationIncrementToggle.checked = rotateClockEverySecond;
  }
}

function initializeRandomFlipSetting() {
  const savedValue = localStorage.getItem(RANDOM_FLIP_STORAGE_KEY);
  randomFlipEverySecond = savedValue === "true";

  if (randomFlipToggle) {
    randomFlipToggle.checked = randomFlipEverySecond;
  }
}

function initializeRotate3DSetting() {
  const savedValue = localStorage.getItem(ROTATE_3D_STORAGE_KEY);
  rotateClock3DEverySecond = savedValue === "true";

  if (rotate3DToggle) {
    rotate3DToggle.checked = rotateClock3DEverySecond;
  }
}

function initializeExtraLineSetting() {
  const savedValue = localStorage.getItem(EXTRA_LINE_STORAGE_KEY);
  showExtraDistractorLine = savedValue === "true";

  if (extraLineToggle) {
    extraLineToggle.checked = showExtraDistractorLine;
  }
}

function initializeArtifactSafeModeSetting() {
  const savedValue = localStorage.getItem(ARTIFACT_SAFE_MODE_STORAGE_KEY);
  artifactSafeMode = savedValue !== "false";

  if (artifactSafeModeToggle) {
    artifactSafeModeToggle.checked = artifactSafeMode;
  }
}

function getThemeByName(themeName) {
  if (themeName === DEFAULT_PLAIN_CLOCK_THEME.name) {
    return DEFAULT_PLAIN_CLOCK_THEME;
  }
  return CLOCK_THEMES.find((theme) => theme.name === themeName) || null;
}

function syncThemeSelectValue() {
  if (!themeSelect || !currentClockTheme || !currentClockTheme.name) {
    return;
  }
  themeSelect.value = currentClockTheme.name;
}

function initializeThemeSelect() {
  if (!themeSelect) {
    return;
  }

  const allThemes = [DEFAULT_PLAIN_CLOCK_THEME, ...CLOCK_THEMES];
  themeSelect.innerHTML = "";

  for (const theme of allThemes) {
    const option = document.createElement("option");
    option.value = theme.name;
    const suffix = theme.era && theme.era !== "classic" ? ` (${theme.era})` : "";
    option.textContent = `${theme.name}${suffix}`;
    themeSelect.appendChild(option);
  }

  syncThemeSelectValue();
}

function persistClockFaceState() {
  const payload = {
    themeName: currentClockTheme && currentClockTheme.name ? currentClockTheme.name : DEFAULT_PLAIN_CLOCK_THEME.name,
    isFlipped,
    clockRotationDegrees,
    clock3DRotationDegrees,
    randomizeThemeEverySecond,
    rotateClockEverySecond,
    randomFlipEverySecond,
    rotateClock3DEverySecond,
    showExtraDistractorLine,
    artifactSafeMode,
  };

  localStorage.setItem(CLOCK_FACE_STATE_STORAGE_KEY, JSON.stringify(payload));
}

function restoreClockFaceState() {
  const raw = localStorage.getItem(CLOCK_FACE_STATE_STORAGE_KEY);
  if (!raw) {
    return false;
  }

  try {
    const parsed = JSON.parse(raw);
    const restoredTheme = getThemeByName(parsed.themeName);
    if (restoredTheme) {
      currentClockTheme = restoredTheme;
    }

    isFlipped = Boolean(parsed.isFlipped);
    clockRotationDegrees = Number.isFinite(Number(parsed.clockRotationDegrees))
      ? Number(parsed.clockRotationDegrees)
      : 0;
    clock3DRotationDegrees = Number.isFinite(Number(parsed.clock3DRotationDegrees))
      ? Number(parsed.clock3DRotationDegrees)
      : 0;

    randomizeThemeEverySecond = Boolean(parsed.randomizeThemeEverySecond);
    rotateClockEverySecond = Boolean(parsed.rotateClockEverySecond);
    randomFlipEverySecond = Boolean(parsed.randomFlipEverySecond);
    rotateClock3DEverySecond = Boolean(parsed.rotateClock3DEverySecond);
    showExtraDistractorLine = parsed.showExtraDistractorLine === true;
    artifactSafeMode = parsed.artifactSafeMode !== false;

    if (randomizeEverySecondToggle) {
      randomizeEverySecondToggle.checked = randomizeThemeEverySecond;
    }
    if (rotationIncrementToggle) {
      rotationIncrementToggle.checked = rotateClockEverySecond;
    }
    if (randomFlipToggle) {
      randomFlipToggle.checked = randomFlipEverySecond;
    }
    if (rotate3DToggle) {
      rotate3DToggle.checked = rotateClock3DEverySecond;
    }
    if (extraLineToggle) {
      extraLineToggle.checked = showExtraDistractorLine;
    }
    if (artifactSafeModeToggle) {
      artifactSafeModeToggle.checked = artifactSafeMode;
    }

    syncThemeSelectValue();

    return true;
  } catch (_err) {
    return false;
  }
}

function persistAcquiredData() {
  try {
    localStorage.setItem(SYMBOLIC_HISTORY_STORAGE_KEY, JSON.stringify(symbolicHistory));
    localStorage.setItem(PROCESSING_HISTORY_STORAGE_KEY, JSON.stringify(processingHistory));
    if (symbolicOutput) {
      localStorage.setItem(SYMBOLIC_OUTPUT_STORAGE_KEY, symbolicOutput.value || "");
    }
    if (probabilisticReasoningOutput) {
      localStorage.setItem(PROBABILISTIC_OUTPUT_STORAGE_KEY, probabilisticReasoningOutput.value || "");
    }
  } catch (_err) {
    // Ignore storage write failures (quota/private mode) and keep runtime state in memory.
  }
}

function normalizeProcessingHistoryEntry(entry) {
  if (!entry || typeof entry !== "object") {
    return null;
  }

  const rawMetrics = extractDebugMetricsFromPayload(entry.metrics)
    || extractDebugMetricsFromPayload(entry)
    || null;

  return {
    logged_at: typeof entry.logged_at === "string" ? entry.logged_at : new Date().toISOString(),
    metrics: normalizeDebugMetrics(rawMetrics || entry.metrics || {}, {}),
    perception_error_seconds: Number.isFinite(Number(entry.perception_error_seconds))
      ? Number(entry.perception_error_seconds)
      : null,
  };
}

function restoreAcquiredData() {
  try {
    const symbolicRaw = localStorage.getItem(SYMBOLIC_HISTORY_STORAGE_KEY);
    if (symbolicRaw) {
      const parsed = JSON.parse(symbolicRaw);
      if (Array.isArray(parsed)) {
        symbolicHistory.splice(0, symbolicHistory.length, ...parsed);
      }
    }

    const processingRaw = localStorage.getItem(PROCESSING_HISTORY_STORAGE_KEY);
    if (processingRaw) {
      const parsed = JSON.parse(processingRaw);
      if (Array.isArray(parsed)) {
        const normalized = parsed
          .slice(-MAX_PROCESSING_HISTORY)
          .map((entry) => normalizeProcessingHistoryEntry(entry))
          .filter((entry) => Boolean(entry));
        processingHistory.splice(0, processingHistory.length, ...normalized);
      }
    }

    const symbolicOutputRaw = localStorage.getItem(SYMBOLIC_OUTPUT_STORAGE_KEY);
    if (symbolicOutput && typeof symbolicOutputRaw === "string" && symbolicOutputRaw.length > 0) {
      symbolicOutput.value = symbolicOutputRaw;
      resizeSymbolicOutput();
    }

    const probabilisticOutputRaw = localStorage.getItem(PROBABILISTIC_OUTPUT_STORAGE_KEY);
    if (probabilisticReasoningOutput && typeof probabilisticOutputRaw === "string" && probabilisticOutputRaw.length > 0) {
      probabilisticReasoningOutput.value = probabilisticOutputRaw;
      resizeProbabilisticReasoningOutput();
    }

    if (symbolicLogOutput && symbolicHistory.length > 0) {
      symbolicLogOutput.value = JSON.stringify(symbolicHistory, null, 2);
      symbolicLogOutput.scrollTop = symbolicLogOutput.scrollHeight;
    }
  } catch (_err) {
    // Ignore malformed persisted data and continue with empty runtime histories.
  }
}

async function clearGeneratedDataOnServer({ preserveExperimentData = false } = {}) {
  const response = await fetch("/clear-generated-data", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      preserve_experiment_data: Boolean(preserveExperimentData),
    }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Failed to clear generated data");
  }

  return data;
}

async function clearWorkspaceData() {
  await clearGeneratedDataOnServer({
    preserveExperimentData: true,
  });
  resetClockFace(true);
  clearLocalDataAndStats({
    statusMessage: "Workspace cleared and clock face reset to defaults. Sweep and LLM benchmark sections were preserved.",
    preserveExperimentSections: true,
  });
}

async function clearDataAndStats({ statusMessage = "All data and stats cleared." } = {}) {
  await clearGeneratedDataOnServer();

  clearLocalDataAndStats({ statusMessage });
}

function clearLocalDataAndStats({
  statusMessage = "All data and stats cleared.",
  preserveExperimentSections = false,
} = {}) {

  // Clear all data arrays
  symbolicHistory.length = 0;
  processingHistory.length = 0;
  lastKnownDebugMetrics = null;
  lastCaptureAt = null;

  // Clear all text output areas
  if (symbolicOutput) {
    symbolicOutput.value = "";
    resizeSymbolicOutput();
  }

  if (probabilisticReasoningOutput) {
    probabilisticReasoningOutput.value = "";
    resizeProbabilisticReasoningOutput();
  }

  if (symbolicLogOutput) {
    symbolicLogOutput.value = "";
    symbolicLogOutput.scrollTop = 0;
  }

  if (!preserveExperimentSections && experimentsSummaryOutput) {
    experimentsSummaryOutput.value = "";
  }

  // Clear processed image and OpenCV stage artifacts from the visible UI.
  resetProcessedArtifactsPanels();

  // Clear grounding script
  clearGroundingScriptSequence();

  if (!preserveExperimentSections) {
    // Reset sweep progress bar
    stopSweepPollTimer();
    sweepProgressValue = 0;
    setSweepProgress(0, "Idle", "running");
    updateSweepStageChecklist("queued", "running");
    updateExperimentsStatus("No experiment run yet.");
    setExperimentsBusyState(false);

  }

  // Clear all localStorage items
  localStorage.removeItem(SYMBOLIC_HISTORY_STORAGE_KEY);
  localStorage.removeItem(PROCESSING_HISTORY_STORAGE_KEY);
  localStorage.removeItem(SYMBOLIC_OUTPUT_STORAGE_KEY);
  localStorage.removeItem(PROBABILISTIC_OUTPUT_STORAGE_KEY);
  localStorage.removeItem(GROUNDING_SCRIPT_SEQUENCE_STORAGE_KEY);
  if (!preserveExperimentSections) {
    localStorage.removeItem(SWEEP_PREVIEWS_STORAGE_KEY);
  }

  // Clear sweep thumbnails
  if (!preserveExperimentSections && sweepConditionPreviews) {
    sweepConditionPreviews.innerHTML = "";
    sweepConditionPreviews.hidden = true;
  }

  // Reset UI displays
  drawProcessingHistogram();
  updateExperimentsStatus(statusMessage);
}

function rearY(value) {
  return showExtraDistractorLine ? value : 0;
}

function drawFaceDistractorLine(radius) {
  if (!ctx || !showExtraDistractorLine) {
    return;
  }

  const angle = (104 * Math.PI) / 180;
  const outer = radius * 0.88;
  const inner = radius * 0.2;
  const xOuter = Math.sin(angle) * outer;
  const yOuter = -Math.cos(angle) * outer;
  const xInner = Math.sin(angle) * inner;
  const yInner = -Math.cos(angle) * inner;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(-xInner, -yInner);
  ctx.lineTo(xOuter, yOuter);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(92, 120, 160, 0.55)";
  ctx.stroke();
  ctx.restore();
}

function toggleTheme() {
  const current = document.body.classList.contains("light-theme") ? "light" : "dark";
  const next = current === "light" ? "dark" : "light";
  localStorage.setItem(THEME_STORAGE_KEY, next);
  applyTheme(next);
}

function autoResizeTextarea(textarea) {
  if (!textarea) {
    return;
  }
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
}

function resizeSymbolicOutput() {
  if (!symbolicOutput) {
    return;
  }

  // Keep symbolic output constrained to the layout column height.
  symbolicOutput.style.height = "100%";
  symbolicOutput.style.minHeight = "0";
  symbolicOutput.style.maxHeight = "none";
}

function resizeProbabilisticReasoningOutput() {
  autoResizeTextarea(probabilisticReasoningOutput);
}

function resizeSymbolicLogOutput() {
  if (!symbolicLogOutput) {
    return;
  }
  symbolicLogOutput.style.height = "360px";
}

function formatProbabilisticReasoning(reasoning) {
  if (!reasoning || typeof reasoning !== "object") {
    return "No probabilistic reasoning payload returned.";
  }

  const formatProb = (value) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(6) : "n/a";
  };

  const lines = [];
  lines.push(`Method: ${String(reasoning.method || "unknown")}`);

  if (reasoning.error) {
    lines.push(`Error: ${String(reasoning.error)}`);
    return lines.join("\n");
  }

  if (Number.isFinite(Number(reasoning.world_count))) {
    lines.push(`World Count: ${Number(reasoning.world_count)}`);
  }

  if (reasoning.domain_sizes && typeof reasoning.domain_sizes === "object") {
    const domain = reasoning.domain_sizes;
    lines.push(
      `Domain Sizes: hour=${String(domain.hour ?? "n/a")}, minute=${String(domain.minute ?? "n/a")}, second=${String(domain.second ?? "n/a")}`,
    );
  }

  if (reasoning.map_assignment && typeof reasoning.map_assignment === "object") {
    const map = reasoning.map_assignment;
    lines.push("MAP Assignment:");
    lines.push(`  time=${String(map.time ?? "n/a")}`);
    lines.push(`  hour=${String(map.hour ?? "n/a")}, minute=${String(map.minute ?? "n/a")}, second=${String(map.second ?? "n/a")}`);
    lines.push(`MAP Probability: ${formatProb(reasoning.map_probability)}`);
  }

  const marginals = reasoning.marginals;
  if (marginals && typeof marginals === "object") {
    lines.push("Marginals (top values):");
    ["hour", "minute", "second"].forEach((name) => {
      const entries = Array.isArray(marginals[name]) ? marginals[name] : [];
      if (entries.length === 0) {
        return;
      }
      lines.push(`  ${name}:`);
      entries.slice(0, 5).forEach((entry) => {
        lines.push(`    ${String(entry.value)} -> ${formatProb(entry.probability)}`);
      });
    });
  }

  if (Array.isArray(reasoning.rules) && reasoning.rules.length > 0) {
    lines.push("Rules (posterior satisfaction mass):");
    reasoning.rules.forEach((rule) => {
      lines.push(
        `  ${String(rule.name || "unnamed")}: mass=${formatProb(rule.satisfied_posterior_mass)}, weight=${formatProb(rule.weight)}, hard=${String(Boolean(rule.hard))}`,
      );
    });
  }

  return lines.join("\n");
}

function resizeGroundingScriptOutput() {
  if (!groundingScriptOutput) {
    return;
  }

  const tallestPayload = buildGroundingScriptPayload(GROUNDING_PRESETS.map((preset) => preset.id));
  const tallestScript = JSON.stringify(tallestPayload, null, 2);
  const lineCount = tallestScript.split("\n").length;
  const lineHeightPx = 18;
  const chromePx = 28;
  const minHeightPx = 220;
  const maxHeightPx = 560;
  const targetHeightPx = Math.max(minHeightPx, Math.min(maxHeightPx, lineCount * lineHeightPx + chromePx));

  groundingScriptOutput.style.height = `${targetHeightPx}px`;
  groundingScriptOutput.style.minHeight = `${targetHeightPx}px`;
  groundingScriptOutput.style.maxHeight = `${targetHeightPx}px`;
}

function syncGroundingToggleUI() {
  if (randomizeEverySecondToggle) {
    randomizeEverySecondToggle.checked = randomizeThemeEverySecond;
  }
  if (rotationIncrementToggle) {
    rotationIncrementToggle.checked = rotateClockEverySecond;
  }
  if (randomFlipToggle) {
    randomFlipToggle.checked = randomFlipEverySecond;
  }
  if (rotate3DToggle) {
    rotate3DToggle.checked = rotateClock3DEverySecond;
  }
  if (extraLineToggle) {
    extraLineToggle.checked = showExtraDistractorLine;
  }
  if (artifactSafeModeToggle) {
    artifactSafeModeToggle.checked = artifactSafeMode;
  }
}

function applyGroundingSettings(settings, { redraw = true, persist = true } = {}) {
  if (!settings || typeof settings !== "object") {
    return;
  }

  if (settings.resetClockFace) {
    resetClockFace(true);
  }

  if (typeof settings.isFlipped === "boolean") {
    isFlipped = settings.isFlipped;
  }
  if (Number.isFinite(settings.clockRotationDegrees)) {
    clockRotationDegrees = Number(settings.clockRotationDegrees);
  }
  if (Number.isFinite(settings.clock3DRotationDegrees)) {
    clock3DRotationDegrees = Number(settings.clock3DRotationDegrees);
  }
  if (typeof settings.randomizeThemeEverySecond === "boolean") {
    randomizeThemeEverySecond = settings.randomizeThemeEverySecond;
  }
  if (typeof settings.rotateClockEverySecond === "boolean") {
    rotateClockEverySecond = settings.rotateClockEverySecond;
  }
  if (typeof settings.randomFlipEverySecond === "boolean") {
    randomFlipEverySecond = settings.randomFlipEverySecond;
  }
  if (typeof settings.rotateClock3DEverySecond === "boolean") {
    rotateClock3DEverySecond = settings.rotateClock3DEverySecond;
  }
  if (typeof settings.showExtraDistractorLine === "boolean") {
    showExtraDistractorLine = settings.showExtraDistractorLine;
  }
  if (typeof settings.artifactSafeMode === "boolean") {
    artifactSafeMode = settings.artifactSafeMode;
  }

  localStorage.setItem(RANDOMIZE_EVERY_SECOND_STORAGE_KEY, String(randomizeThemeEverySecond));
  localStorage.setItem(ROTATION_INCREMENT_STORAGE_KEY, String(rotateClockEverySecond));
  localStorage.setItem(RANDOM_FLIP_STORAGE_KEY, String(randomFlipEverySecond));
  localStorage.setItem(ROTATE_3D_STORAGE_KEY, String(rotateClock3DEverySecond));
  localStorage.setItem(EXTRA_LINE_STORAGE_KEY, String(showExtraDistractorLine));
  localStorage.setItem(ARTIFACT_SAFE_MODE_STORAGE_KEY, String(artifactSafeMode));

  syncGroundingToggleUI();
  if (redraw) {
    drawClock();
  }
  if (persist) {
    persistClockFaceState();
  }
}

function persistGroundingScriptSequence() {
  localStorage.setItem(GROUNDING_SCRIPT_SEQUENCE_STORAGE_KEY, JSON.stringify(groundingScriptSequence));
}

function restoreGroundingScriptSequence() {
  const raw = localStorage.getItem(GROUNDING_SCRIPT_SEQUENCE_STORAGE_KEY);
  if (!raw) {
    groundingScriptSequence = [];
    return;
  }

  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      groundingScriptSequence = parsed.filter((id) => groundingPresetById.has(id));
    }
  } catch (_err) {
    groundingScriptSequence = [];
  }
}

function buildGroundingScriptPayload(sequence = groundingScriptSequence) {
  const steps = sequence
    .map((id) => groundingPresetById.get(id))
    .filter(Boolean)
    .map((preset, index) => ({
      step: index + 1,
      id: preset.id,
      title: preset.title,
      summary: preset.summary,
      settings: preset.settings,
    }));

  const resolvedSettings = {};
  for (const step of steps) {
    Object.assign(resolvedSettings, step.settings);
  }

  return {
    script_type: "grounding",
    version: "1.0",
    generated_at: new Date().toISOString(),
    steps,
    resolved_settings: resolvedSettings,
  };
}

function updateGroundingScriptOutput() {
  if (!groundingScriptOutput) {
    return;
  }

  const payload = buildGroundingScriptPayload();
  if (payload.steps.length === 0) {
    groundingScriptOutput.value = "Add presets to the Script Generator to build a grounding script.";
    resizeGroundingScriptOutput();
    return;
  }

  groundingScriptOutput.value = JSON.stringify(payload, null, 2);
  resizeGroundingScriptOutput();
}

function renderGroundingPresetLibrary() {
  if (!groundingPresetLibrary) {
    return;
  }

  groundingPresetLibrary.innerHTML = "";
  for (const preset of GROUNDING_PRESETS) {
    const alreadyAdded = groundingScriptSequence.includes(preset.id);
    const card = document.createElement("div");
    card.className = alreadyAdded ? "grounding-card grounding-card-disabled" : "grounding-card";
    card.draggable = !alreadyAdded;
    card.dataset.presetId = preset.id;
    card.setAttribute("aria-disabled", alreadyAdded ? "true" : "false");

    const title = document.createElement("p");
    title.className = "grounding-card-title";
    title.textContent = preset.title;

    const summary = document.createElement("p");
    summary.className = "grounding-card-summary";
    summary.textContent = preset.summary;

    card.appendChild(title);
    card.appendChild(summary);

    card.addEventListener("dragstart", (event) => {
      if (alreadyAdded) {
        event.preventDefault();
        return;
      }
      if (!event.dataTransfer) {
        return;
      }
      event.dataTransfer.setData("text/plain", preset.id);
      event.dataTransfer.effectAllowed = "copy";
    });

    groundingPresetLibrary.appendChild(card);
  }
}

function renderGroundingScriptSequence() {
  if (!groundingScriptSequenceEl) {
    return;
  }

  groundingScriptSequenceEl.innerHTML = "";
  groundingScriptSequence.forEach((presetId, idx) => {
    const preset = groundingPresetById.get(presetId);
    if (!preset) {
      return;
    }

    const row = document.createElement("div");
    row.className = "grounding-sequence-item";

    const text = document.createElement("div");
    const title = document.createElement("p");
    title.className = "grounding-card-title";
    title.textContent = `${idx + 1}. ${preset.title}`;
    const summary = document.createElement("p");
    summary.textContent = preset.summary;
    text.appendChild(title);
    text.appendChild(summary);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "grounding-remove-btn";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => {
      groundingScriptSequence.splice(idx, 1);
      persistGroundingScriptSequence();
      renderGroundingScriptSequence();
      renderGroundingPresetLibrary();
      updateGroundingScriptOutput();
    });

    row.appendChild(text);
    row.appendChild(removeBtn);
    groundingScriptSequenceEl.appendChild(row);
  });
}

function addGroundingPresetToSequence(presetId) {
  if (!groundingPresetById.has(presetId)) {
    return;
  }

  if (groundingScriptSequence.includes(presetId)) {
    if (symbolicOutput) {
      symbolicOutput.value = "That grounding preset is already in the script sequence.";
      resizeSymbolicOutput();
    }
    return;
  }

  groundingScriptSequence.push(presetId);
  persistGroundingScriptSequence();
  renderGroundingScriptSequence();
  renderGroundingPresetLibrary();
  updateGroundingScriptOutput();
}

function addAllGroundingPresetsToSequence() {
  const missing = GROUNDING_PRESETS.map((preset) => preset.id).filter((id) => !groundingScriptSequence.includes(id));
  if (missing.length === 0) {
    if (symbolicOutput) {
      symbolicOutput.value = "All grounding presets are already present in the script sequence.";
      resizeSymbolicOutput();
    }
    return;
  }

  groundingScriptSequence.push(...missing);
  persistGroundingScriptSequence();
  renderGroundingScriptSequence();
  renderGroundingPresetLibrary();
  updateGroundingScriptOutput();
}

function initializeGroundingScriptDropzone() {
  if (!groundingScriptDropzone) {
    return;
  }

  groundingScriptDropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    groundingScriptDropzone.classList.add("drag-over");
  });

  groundingScriptDropzone.addEventListener("dragleave", () => {
    groundingScriptDropzone.classList.remove("drag-over");
  });

  groundingScriptDropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    groundingScriptDropzone.classList.remove("drag-over");

    const presetId = event.dataTransfer ? event.dataTransfer.getData("text/plain") : "";
    addGroundingPresetToSequence(presetId);
  });
}

function applyGroundingScriptSequence() {
  const payload = buildGroundingScriptPayload();
  if (!payload.steps.length) {
    if (symbolicOutput) {
      symbolicOutput.value = "Grounding script is empty. Drag presets into the Script Generator first.";
      resizeSymbolicOutput();
    }
    return;
  }

  for (const step of payload.steps) {
    applyGroundingSettings(step.settings, { redraw: false, persist: false });
  }

  drawClock();
  persistClockFaceState();

  if (symbolicOutput) {
    symbolicOutput.value = `Applied grounding script with ${payload.steps.length} preset step(s).`;
    resizeSymbolicOutput();
  }
}

async function copyGroundingScriptToClipboard() {
  if (!groundingScriptOutput) {
    return;
  }

  const text = groundingScriptOutput.value || "";
  if (!text.trim()) {
    return;
  }

  await navigator.clipboard.writeText(text);
  if (symbolicOutput) {
    symbolicOutput.value = "Copied generated grounding script to clipboard.";
    resizeSymbolicOutput();
  }
}

function clearGroundingScriptSequence() {
  groundingScriptSequence = [];
  persistGroundingScriptSequence();
  renderGroundingScriptSequence();
  renderGroundingPresetLibrary();
  updateGroundingScriptOutput();
}

function initializeGroundingPresetStudio() {
  restoreGroundingScriptSequence();
  renderGroundingPresetLibrary();
  initializeGroundingScriptDropzone();
  renderGroundingScriptSequence();
  updateGroundingScriptOutput();
}

function updateExperimentsStatus(message) {
  if (!experimentsStatus) {
    return;
  }
  experimentsStatus.textContent = message;
}

function setExperimentsBusyState(isBusy) {
  experimentsBusy = isBusy;
  if (runExperimentsBtn) {
    runExperimentsBtn.disabled = isBusy;
  }
  if (refreshExperimentsBtn) {
    refreshExperimentsBtn.disabled = isBusy;
  }
}

function setExperimentsSummaryText(text) {
  if (!experimentsSummaryOutput) {
    return;
  }
  experimentsSummaryOutput.value = text || "";
}

function clearExperimentSectionUi() {
  setExperimentsSummaryText("");
  stopSweepPollTimer();
  sweepProgressValue = 0;
  setSweepProgress(0, "Idle", "running");
  updateSweepStageChecklist("queued", "running");
  updateExperimentsStatus("No experiment run yet.");
  setExperimentsBusyState(false);
  if (sweepConditionPreviews) {
    sweepConditionPreviews.innerHTML = "";
    sweepConditionPreviews.hidden = true;
  }
  localStorage.removeItem(SWEEP_PREVIEWS_STORAGE_KEY);
}

async function clearExperimentSectionData() {
  const response = await fetch("/clear-experiment-data", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Failed to clear experiment sweep data");
  }

  clearExperimentSectionUi();
  return data;
}

function setSweepProgress(percent, label, state = "running") {
  sweepProgressValue = Number.isFinite(Number(percent)) ? Math.max(0, Math.min(100, Number(percent))) : 0;

  if (sweepProgressFill) {
    sweepProgressFill.style.width = `${sweepProgressValue.toFixed(1)}%`;
  }

  if (sweepProgressLabel) {
    sweepProgressLabel.textContent = label || `${Math.floor(sweepProgressValue)}%`;
  }

  if (sweepProgressTrack) {
    sweepProgressTrack.setAttribute("aria-valuenow", String(Math.floor(sweepProgressValue)));
  }

  if (sweepProgressWrap) {
    sweepProgressWrap.classList.remove("is-complete", "is-error");
    if (state === "complete") {
      sweepProgressWrap.classList.add("is-complete");
    } else if (state === "error") {
      sweepProgressWrap.classList.add("is-error");
    }
  }
}

function updateSweepStageChecklist(currentStageRaw, state = "running") {
  if (!sweepStageList) {
    return;
  }

  sweepStageList.classList.remove("is-error");
  if (state === "error") {
    sweepStageList.classList.add("is-error");
  }

  const stageOrder = ["queued", "running", "stats", "done"];
  let currentKey;
  if (state === "complete") {
    currentKey = "done";
  } else {
    const raw = String(currentStageRaw || "").toLowerCase();
    if (raw.includes("queue")) {
      currentKey = "queued";
    } else if (raw.includes("stat")) {
      currentKey = "stats";
    } else if (raw.includes("complete") || raw.includes("done")) {
      currentKey = "done";
    } else {
      currentKey = "running";
    }
  }

  const currentIdx = stageOrder.indexOf(currentKey);
  const chips = sweepStageList.querySelectorAll(".llm-stage-chip");
  chips.forEach((chip) => {
    chip.classList.remove("is-active", "is-complete");
    const key = chip.getAttribute("data-stage-key") || "";
    const idx = stageOrder.indexOf(key);
    if (idx !== -1 && idx < currentIdx) {
      chip.classList.add("is-complete");
    }
    if (key === currentKey) {
      chip.classList.add(state === "complete" ? "is-complete" : "is-active");
    }
  });
}

function stopSweepPollTimer() {
  if (sweepPollTimer !== null) {
    clearInterval(sweepPollTimer);
    sweepPollTimer = null;
  }
}

function renderSweepPreviews(conditions) {
  if (!sweepConditionPreviews || !conditions.length) {
    return;
  }
  sweepConditionPreviews.innerHTML = conditions
    .map(
      (c) =>
        `<figure class="sweep-thumbnail" role="button" tabindex="0" aria-label="Expand ${c.condition_id}"
          data-img="${c.imageData}" data-id="${c.condition_id}" data-desc="${c.description}" data-severity="${c.severity}" data-family="${c.family}">
          <img src="${c.imageData}" alt="${c.condition_id}" width="96" height="96" loading="lazy" />
          <figcaption>
            <span class="sweep-thumb-id">${c.condition_id}</span>
            <span class="sweep-thumb-severity sweep-thumb-severity--${c.severity}">${c.severity}</span>
          </figcaption>
        </figure>`
    )
    .join("");
  sweepConditionPreviews.hidden = false;
  initSweepLightbox();
}

function restoreSweepPreviewsFromStorage() {
  try {
    const raw = localStorage.getItem(SWEEP_PREVIEWS_STORAGE_KEY);
    if (!raw) {
      return;
    }
    const conditions = JSON.parse(raw);
    if (Array.isArray(conditions) && conditions.length) {
      renderSweepPreviews(conditions);
    }
  } catch (_) {
    // stale or corrupt cache — ignore
  }
}

function persistAndRenderSweepPreviews(conditions) {
  if (!Array.isArray(conditions) || !conditions.length) {
    return false;
  }
  try {
    localStorage.setItem(SWEEP_PREVIEWS_STORAGE_KEY, JSON.stringify(conditions));
  } catch (_) {
    // storage quota exceeded — thumbnails will still show this session
  }
  renderSweepPreviews(conditions);
  return true;
}

async function loadSweepConditionPreviews() {
  if (!sweepConditionPreviews) {
    return;
  }
  try {
    const response = await fetch("/sweep-condition-previews");
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    const conditions = Array.isArray(data.conditions) ? data.conditions : [];
    if (!conditions.length) {
      return;
    }
    persistAndRenderSweepPreviews(conditions);
  } catch (_) {
    // previews are non-critical; silently ignore
  }
}

function initSweepLightbox() {
  const lightbox = document.getElementById("sweepLightbox");
  const lightboxImg = document.getElementById("sweepLightboxImg");
  const lightboxCaption = document.getElementById("sweepLightboxCaption");
  const closeBtn = document.getElementById("sweepLightboxClose");
  if (!lightbox || !lightboxImg || !lightboxCaption || !closeBtn) {
    return;
  }

  function openLightbox(fig) {
    lightboxImg.src = fig.dataset.img;
    lightboxImg.alt = fig.dataset.id;
    lightboxCaption.innerHTML =
      `<strong>${fig.dataset.id}</strong> &mdash; ${fig.dataset.desc}` +
      `<span class="sweep-thumb-severity sweep-thumb-severity--${fig.dataset.severity}" style="margin-left:8px">${fig.dataset.severity}</span>`;
    lightbox.hidden = false;
    lightbox.focus();
  }

  function closeLightbox() {
    lightbox.hidden = true;
    lightboxImg.src = "";
  }

  // Delegated click on the thumbnails container
  sweepConditionPreviews.addEventListener("click", (e) => {
    const fig = e.target.closest(".sweep-thumbnail");
    if (fig) {
      openLightbox(fig);
    }
  });

  sweepConditionPreviews.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      const fig = e.target.closest(".sweep-thumbnail");
      if (fig) {
        e.preventDefault();
        openLightbox(fig);
      }
    }
  });

  closeBtn.addEventListener("click", closeLightbox);

  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) {
      closeLightbox();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) {
      closeLightbox();
    }
  });
}

async function pollSweepProgress() {
  try {
    const response = await fetch("/sweep-progress");
    const state = await response.json();
    if (!response.ok) {
      throw new Error(state.error || "Failed to read sweep progress");
    }

    const percent = Number.isFinite(Number(state.percent)) ? Number(state.percent) : 0;
    const running = Boolean(state.running);
    const completed = Boolean(state.completed);
    const failed = Boolean(state.failed);
    const stage = String(state.stage || "running");

    if (running) {
      const label = `${stage} ${Math.floor(percent)}%`;
      setSweepProgress(percent, label, "running");
      updateSweepStageChecklist(stage, "running");
      return;
    }

    if (completed) {
      stopSweepPollTimer();
      setSweepProgress(100, "Completed", "complete");
      updateSweepStageChecklist("done", "complete");
      if (typeof state.summaryPreview === "string") {
        setExperimentsSummaryText(state.summaryPreview);
      }
      updateExperimentsStatus(
        `Sweep completed. Metrics and stats saved in results/.`
      );
      setExperimentsBusyState(false);
      return;
    }

    if (failed) {
      stopSweepPollTimer();
      const failedAt = Math.max(5, Math.floor(percent || sweepProgressValue));
      setSweepProgress(failedAt, `Failed at ${failedAt}%`, "error");
      updateSweepStageChecklist(stage, "error");
      updateExperimentsStatus(String(state.error || "Experiment sweep failed."));
      setExperimentsBusyState(false);
    }
  } catch (err) {
    stopSweepPollTimer();
    updateExperimentsStatus(String(err && err.message ? err.message : err));
    setSweepProgress(Math.max(5, Math.floor(sweepProgressValue)), "Progress unavailable", "error");
    updateSweepStageChecklist("error", "error");
    setExperimentsBusyState(false);
  }
}

function startSweepProgressPolling() {
  stopSweepPollTimer();
  setSweepProgress(2, "Queued 2%", "running");
  updateSweepStageChecklist("queued", "running");
  void pollSweepProgress();
  sweepPollTimer = setInterval(() => {
    void pollSweepProgress();
  }, 1200);
}


async function refreshExperimentsSummary() {
  try {
    const [summaryResponse, artifactsResponse] = await Promise.all([
      fetch("/experiment-summary"),
      fetch("/experiment-artifacts"),
    ]);

    const summaryData = await summaryResponse.json();
    const artifactsData = await artifactsResponse.json();

    if (!summaryResponse.ok) {
      throw new Error(summaryData.error || "Failed to load experiment summary");
    }
    if (!artifactsResponse.ok) {
      throw new Error(artifactsData.error || "Failed to load artifact list");
    }

    const summaryText = typeof summaryData.summary === "string" ? summaryData.summary : "No summary available.";
    setExperimentsSummaryText(summaryText);

    const existing = Array.isArray(artifactsData.artifacts)
      ? artifactsData.artifacts.filter((a) => a && a.exists).map((a) => a.name)
      : [];
    updateExperimentsStatus(
      existing.length > 0
        ? `Artifacts ready: ${existing.join(", ")}`
        : "No experiment artifacts yet. Run a sweep first."
    );
  } catch (err) {
    updateExperimentsStatus(String(err && err.message ? err.message : err));
  }
}

// Password verification helper
async function promisePassword(actionName) {
  return new Promise((resolve) => {
    const modal = document.getElementById("passwordModal");
    const title = document.getElementById("passwordModalTitle");
    const input = document.getElementById("passwordInput");
    const submitBtn = document.getElementById("passwordModalSubmit");
    const cancelBtn = document.getElementById("passwordModalCancel");

    if (!modal || !title || !input || !submitBtn || !cancelBtn) {
      resolve(null);
      return;
    }

    title.textContent = `Enter password to ${actionName}`;
    input.value = "";
    modal.style.display = "flex";
    input.focus();

    const handleSubmit = () => {
      const password = input.value;
      cleanup();
      resolve(password);
    };

    const handleCancel = () => {
      cleanup();
      resolve(null);
    };

    const handleKeyPress = (e) => {
      if (e.key === "Enter") {
        handleSubmit();
      } else if (e.key === "Escape") {
        handleCancel();
      }
    };

    const cleanup = () => {
      modal.style.display = "none";
      submitBtn.removeEventListener("click", handleSubmit);
      cancelBtn.removeEventListener("click", handleCancel);
      input.removeEventListener("keypress", handleKeyPress);
    };

    submitBtn.addEventListener("click", handleSubmit);
    cancelBtn.addEventListener("click", handleCancel);
    input.addEventListener("keypress", handleKeyPress);
  });
}

async function verifyPassword(password) {
  try {
    const response = await fetch("/verify-password", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ password }),
    });
    const data = await response.json();
    return data.valid === true;
  } catch (err) {
    console.error("Password verification error:", err);
    return false;
  }
}

async function runExperimentSweepFromUI() {
  const samplesPerCondition = Number.parseInt(expSamplesInput ? expSamplesInput.value : "40", 10);
  const statsIterations = Number.parseInt(expStatsIterationsInput ? expStatsIterationsInput.value : "2000", 10);

  if (!Number.isFinite(samplesPerCondition) || samplesPerCondition < 1 || samplesPerCondition > 500) {
    updateExperimentsStatus("Samples/condition must be between 1 and 500.");
    return;
  }

  if (!Number.isFinite(statsIterations) || statsIterations < 100 || statsIterations > 50000) {
    updateExperimentsStatus("Stats iterations must be between 100 and 50000.");
    return;
  }

  if (isPasswordPromptRequired("run_experiments")) {
    const password = await promisePassword("run experiments");
    if (!password) {
      updateExperimentsStatus("Experiment run cancelled.");
      return;
    }

    const isValid = await verifyPassword(password);
    if (!isValid) {
      updateExperimentsStatus("Invalid password. Experiment run cancelled.");
      return;
    }
  }

  setExperimentsBusyState(true);
  updateExperimentsStatus("Starting experiment sweep...");

  try {
    const response = await fetch("/run-experiments", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        samplesPerCondition,
        statsIterations,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Experiment run failed");
    }

    updateExperimentsStatus("Running experiment sweep. This may take a while...");
    const renderedFromStartResponse = persistAndRenderSweepPreviews(data.conditions);
    if (!renderedFromStartResponse) {
      void loadSweepConditionPreviews();
    }
    startSweepProgressPolling();
  } catch (err) {
    updateExperimentsStatus(String(err && err.message ? err.message : err));
    setSweepProgress(0, "Idle");
    setExperimentsBusyState(false);
  }
}

function downloadExperimentArtifact(artifactName) {
  window.open(`/experiment-artifact/${encodeURIComponent(artifactName)}`, "_blank", "noopener,noreferrer");
}

function appendSymbolicLog(symbolicData) {
  const entry = {
    logged_at: new Date().toISOString(),
    capture_at: lastCaptureAt ? lastCaptureAt.toISOString() : null,
    symbolic: symbolicData,
  };
  symbolicHistory.push(entry);
  if (symbolicHistory.length > MAX_SYMBOLIC_HISTORY) {
    symbolicHistory.shift();
  }

  if (symbolicLogOutput) {
    symbolicLogOutput.value = JSON.stringify(symbolicHistory, null, 2);
    symbolicLogOutput.scrollTop = symbolicLogOutput.scrollHeight;
  }

  persistAcquiredData();
}

function appendSymbolicFailureLog(errorMessage) {
  const entry = {
    logged_at: new Date().toISOString(),
    capture_at: lastCaptureAt ? lastCaptureAt.toISOString() : null,
    status: "failed_nonfatal",
    error: errorMessage,
  };
  symbolicHistory.push(entry);
  if (symbolicHistory.length > MAX_SYMBOLIC_HISTORY) {
    symbolicHistory.shift();
  }

  if (symbolicLogOutput) {
    symbolicLogOutput.value = JSON.stringify(symbolicHistory, null, 2);
    symbolicLogOutput.scrollTop = symbolicLogOutput.scrollHeight;
  }

  persistAcquiredData();
}

function isNonFatalRealtimeError(err) {
  const message = String(err && err.message ? err.message : err);
  return message.includes(NON_FATAL_HAND_DETECTION_ERROR_TEXT);
}

function extractDebugMetricsFromPayload(payload) {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  if (payload.debug_metrics && typeof payload.debug_metrics === "object") {
    return payload.debug_metrics;
  }

  if (payload.symbolic && typeof payload.symbolic === "object") {
    const nestedMetrics = payload.symbolic.debug_metrics;
    if (nestedMetrics && typeof nestedMetrics === "object") {
      return nestedMetrics;
    }
  }

  return null;
}

function readNumericMetric(metrics, keys) {
  for (const key of keys) {
    const value = Number(metrics && metrics[key]);
    if (Number.isFinite(value)) {
      return value;
    }
  }
  return 0;
}

function normalizeDebugMetrics(metrics = {}, { forceFallback = false, fallbackMode = null } = {}) {
  const candidateAngles = Array.isArray(metrics.candidate_angles_deg)
    ? metrics.candidate_angles_deg.filter((v) => Number.isFinite(Number(v))).map((v) => Number(v))
    : [];

  return {
    pipeline: typeof metrics.pipeline === "string" ? metrics.pipeline : "color-mask + hough-fallback",
    fallback_used: forceFallback ? true : Boolean(metrics.fallback_used),
    fallback_mode: fallbackMode || metrics.fallback_mode || "none",
    candidate_count: Number.isFinite(Number(metrics.candidate_count))
      ? Number(metrics.candidate_count)
      : candidateAngles.length,
    candidate_angles_deg: candidateAngles,
    hour_mask_pixels: readNumericMetric(metrics, ["hour_mask_pixels", "hourMaskPixels", "hour_component_pixels", "hourComponentPixels"]),
    minute_mask_pixels: readNumericMetric(metrics, ["minute_mask_pixels", "minuteMaskPixels", "minute_component_pixels", "minuteComponentPixels"]),
    second_mask_pixels: readNumericMetric(metrics, ["second_mask_pixels", "secondMaskPixels"]),
    hour_component_pixels: readNumericMetric(metrics, ["hour_component_pixels", "hourComponentPixels"]),
    minute_component_pixels: readNumericMetric(metrics, ["minute_component_pixels", "minuteComponentPixels"]),
    color_hour_detected: Boolean(metrics.color_hour_detected),
    color_minute_detected: Boolean(metrics.color_minute_detected),
    color_second_detected: Boolean(metrics.color_second_detected),
  };
}

function appendProcessingMetrics(symbolicData, { fromRealtimeFailure = false } = {}) {
  const rawMetrics = extractDebugMetricsFromPayload(symbolicData);
  if (!rawMetrics && !fromRealtimeFailure) {
    return;
  }

  const metricsBase = rawMetrics || lastKnownDebugMetrics || {};
  const metrics = normalizeDebugMetrics(metricsBase, {
    forceFallback: fromRealtimeFailure,
    fallbackMode: fromRealtimeFailure ? "realtime-nonfatal-placeholder" : null,
  });
  lastKnownDebugMetrics = metrics;

  let perceptionErrorSeconds = null;
  if (lastCaptureAt instanceof Date && !Number.isNaN(lastCaptureAt.getTime())) {
    const hour12 = Number.parseInt(symbolicData && symbolicData.HOUR_HAND, 10);
    const minute = Number.parseInt(symbolicData && symbolicData.MINUTE_HAND, 10);
    const second = Number.parseInt(symbolicData && symbolicData.SECOND_HAND, 10);

    if (Number.isFinite(hour12) && Number.isFinite(minute) && Number.isFinite(second)) {
      const normalizedHour = ((hour12 % 12) + 12) % 12;
      const symbolicSeconds = normalizedHour * 3600 + (minute % 60) * 60 + (second % 60);

      const captureHour12 = lastCaptureAt.getHours() % 12;
      const captureSeconds = captureHour12 * 3600 + lastCaptureAt.getMinutes() * 60 + lastCaptureAt.getSeconds();

      const rawDiff = Math.abs(symbolicSeconds - captureSeconds);
      perceptionErrorSeconds = Math.min(rawDiff, (12 * 3600) - rawDiff);
    }
  }

  processingHistory.push(normalizeProcessingHistoryEntry({
    logged_at: new Date().toISOString(),
    metrics,
    perception_error_seconds: perceptionErrorSeconds,
  }));

  if (processingHistory.length > MAX_PROCESSING_HISTORY) {
    processingHistory.shift();
  }

  persistAcquiredData();

  drawProcessingHistogram();
}

function drawProcessingHistogramNow() {
  if (!histogramCtx || !histogramCanvas) {
    return;
  }

  const cssWidth = Math.max(1, Math.round(histogramCanvas.clientWidth || histogramCanvas.width));
  const cssHeight = Math.max(1, Math.round(histogramCanvas.clientHeight || histogramCanvas.height));
  const dpr = window.devicePixelRatio || 1;
  const targetWidth = Math.max(1, Math.round(cssWidth * dpr));
  const targetHeight = Math.max(1, Math.round(cssHeight * dpr));

  if (histogramCanvas.width !== targetWidth || histogramCanvas.height !== targetHeight) {
    histogramCanvas.width = targetWidth;
    histogramCanvas.height = targetHeight;
  }

  histogramCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = cssWidth;
  const h = cssHeight;
  const isLight = document.body.classList.contains("light-theme");
  const ink = isLight ? "#12324f" : "#dce8f7";
  const grid = isLight ? "#d2e6f8" : "#2a4967";
  const bar = isLight ? "#1f5f8b" : "#3d84bf";
  const legendBg = isLight ? "#e8f3ff" : "#1a3048";
  const legendBorder = isLight ? "#a9c9e6" : "#3a5d7f";
  const canvasBg = isLight ? "#f7fbff" : "#13253a";
  const chartBg = isLight ? "#f3f9ff" : "#162b42";

  histogramCtx.clearRect(0, 0, w, h);
  histogramCtx.fillStyle = canvasBg;
  histogramCtx.fillRect(0, 0, w, h);

  histogramCtx.fillStyle = ink;
  histogramCtx.font = "600 15px 'Segoe UI', sans-serif";
  histogramCtx.fillText("Live Processing Overview", 16, 24);
  histogramCtx.font = "500 12px 'Segoe UI', sans-serif";
  histogramCtx.fillText("Angles, mask strength, and symbolic error trend", 16, 40);

  const drawNoTelemetryMessage = () => {
    histogramCtx.font = "600 14px 'Segoe UI', sans-serif";
    histogramCtx.fillText("No symbolic telemetry yet. Convert once to start plotting.", 16, 68);
  };

  const drawPanel = (x, y, width, height) => {
    histogramCtx.fillStyle = chartBg;
    histogramCtx.fillRect(x, y, width, height);
    histogramCtx.strokeStyle = legendBorder;
    histogramCtx.lineWidth = 1;
    histogramCtx.strokeRect(x, y, width, height);
  };

  if (processingHistory.length === 0) {
    drawNoTelemetryMessage();
    return;
  }

  const recent = processingHistory
    .slice(-40)
    .filter((entry) => entry && typeof entry === "object" && entry.metrics && typeof entry.metrics === "object");
  if (recent.length === 0) {
    drawNoTelemetryMessage();
    return;
  }
  const bins = Array.from({ length: 12 }, () => 0);
  for (const item of recent) {
    const angles = Array.isArray(item.metrics.candidate_angles_deg) ? item.metrics.candidate_angles_deg : [];
    for (const angle of angles) {
      const normalized = ((Number(angle) % 360) + 360) % 360;
      const idx = Math.floor(normalized / 30) % 12;
      bins[idx] += 1;
    }
  }

  const mainX = 16;
  const mainY = 52;
  const mainW = w - 32;
  const mainH = 170;
  const mainBottom = mainY + mainH;
  const barWidth = mainW / bins.length;
  const maxBin = Math.max(1, ...bins);

  drawPanel(mainX, mainY, mainW, mainH);

  histogramCtx.fillStyle = ink;
  histogramCtx.font = "bold 13px Trebuchet MS";
  histogramCtx.fillText("Candidate Angle Histogram", mainX + 8, mainY + 16);

  histogramCtx.strokeStyle = grid;
  histogramCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = mainY + 24 + ((mainH - 34) * i) / 4;
    histogramCtx.beginPath();
    histogramCtx.moveTo(mainX + 6, y);
    histogramCtx.lineTo(mainX + mainW - 6, y);
    histogramCtx.stroke();
  }

  histogramCtx.fillStyle = bar;
  bins.forEach((count, idx) => {
    const x = mainX + idx * barWidth + 4;
    const hNorm = count / maxBin;
    const usableH = mainH - 36;
    const barH = hNorm * usableH;
    const y = mainBottom - 8 - barH;
    histogramCtx.fillRect(x, y, Math.max(4, barWidth - 8), barH);
  });

  histogramCtx.fillStyle = ink;
  histogramCtx.font = "11px Courier New";
  bins.forEach((_, idx) => {
    const label = String(idx * 30);
    const x = mainX + idx * barWidth + 4;
    histogramCtx.fillText(label, x, mainBottom + 14);
  });

  const avg = (key) => {
    let sum = 0;
    let n = 0;
    for (const item of recent) {
      const value = Number(item.metrics[key]);
      if (Number.isFinite(value)) {
        sum += value;
        n += 1;
      }
    }
    return n === 0 ? 0 : sum / n;
  };

  const recentMaskWindow = recent.slice(-8);
  const recentAvg = (key) => {
    let sum = 0;
    let n = 0;
    for (const item of recentMaskWindow) {
      const value = Number(item.metrics[key]);
      if (Number.isFinite(value)) {
        sum += value;
        n += 1;
      }
    }
    return n === 0 ? 0 : sum / n;
  };

  const latestMask = (key) => {
    const latest = recent[recent.length - 1];
    const value = Number(latest && latest.metrics ? latest.metrics[key] : NaN);
    return Number.isFinite(value) ? value : 0;
  };

  const maskValues = [
    { label: "HOUR MASK PX", key: "hour_mask_pixels", color: isLight ? "#1b6fa8" : "#69c2ff" },
    { label: "MIN MASK PX", key: "minute_mask_pixels", color: isLight ? "#2a8ccc" : "#8cd7ff" },
    { label: "SEC MASK PX", key: "second_mask_pixels", color: "#c32828" },
  ].map((m) => {
    const longAvg = avg(m.key);
    const shortAvg = recentAvg(m.key);
    const latest = latestMask(m.key);
    // Weighted toward most recent samples so realtime changes are visible.
    const responsiveValue = (0.5 * latest) + (0.35 * shortAvg) + (0.15 * longAvg);
    return { ...m, value: responsiveValue };
  });
  const realtimeFailureCount = recent.filter(
    (item) => item.metrics.fallback_mode === "realtime-nonfatal-placeholder"
  ).length;
  const actualFallbackCount = recent.filter(
    (item) => Boolean(item.metrics.fallback_used) && item.metrics.fallback_mode !== "realtime-nonfatal-placeholder"
  ).length;
  const hasAnyMaskSignal = maskValues.some((item) => item.value > 0);
  let maskStatusText = "Color-mask signal is present.";
  if (!hasAnyMaskSignal) {
    if (realtimeFailureCount === recent.length && recent.length > 0) {
      maskStatusText = "Realtime frames are mostly non-fatal failures, so fresh mask metrics are not arriving.";
    } else if (actualFallbackCount === recent.length && recent.length > 0) {
      maskStatusText = "Color-mask detection is weak and fallback is dominating successful frames.";
    } else {
      maskStatusText = "No mask metrics received yet for the recent frames.";
    }
  }

  const maxMask = Math.max(1, ...maskValues.map((m) => m.value));
  const lowerY = 246;
  const panelGap = 12;
  const panelW = Math.floor((w - 32 - panelGap) / 2);
  const panelH = 126;

  const maskX = 16;
  const maskY = lowerY;
  drawPanel(maskX, maskY, panelW, panelH);

  histogramCtx.fillStyle = ink;
  histogramCtx.font = "bold 12px Trebuchet MS";
  histogramCtx.fillText("Average Mask Strength", maskX + 8, maskY + 16);

  const slotW = (panelW - 24) / 3;
  const barsBaseY = maskY + 78;
  const maxBar = 40;
  maskValues.forEach((m, idx) => {
    const slotX = maskX + 10 + idx * slotW;
    const hNorm = m.value / maxMask;
    const rawBarH = hNorm * maxBar;
    const barH = m.value > 0 ? Math.max(3, rawBarH) : 0;

    histogramCtx.fillStyle = m.color;
    histogramCtx.fillRect(slotX + slotW * 0.25, barsBaseY - barH, slotW * 0.5, barH);

    histogramCtx.fillStyle = ink;
    histogramCtx.font = "10px Courier New";
    histogramCtx.fillText(m.label.replace(" MASK PX", ""), slotX + 2, maskY + 96);
    histogramCtx.font = "bold 11px Courier New";
    histogramCtx.fillText(String(Math.round(m.value)), slotX + 2, maskY + 112);
  });

  histogramCtx.fillStyle = ink;
  histogramCtx.font = "10px Trebuchet MS";
  histogramCtx.fillText(maskStatusText, maskX + 8, maskY + 32);

  const fallbackCount = actualFallbackCount;

  const errorSeries = recent
    .map((item) => Number(item.perception_error_seconds))
    .filter((value) => Number.isFinite(value));

  const lineLeft = maskX + panelW + panelGap;
  const lineTop = lowerY;
  const lineWidth = w - lineLeft - 16;
  const lineHeight = panelH;

  drawPanel(lineLeft, lineTop, lineWidth, lineHeight);

  histogramCtx.fillStyle = ink;
  histogramCtx.font = "bold 12px Trebuchet MS";
  histogramCtx.fillText("Perception Error Trend (sec)", lineLeft + 8, lineTop + 16);

  const plotTop = lineTop + 22;
  const plotBottom = lineTop + lineHeight - 24;
  const plotHeight = Math.max(8, plotBottom - plotTop);
  const metricsRowY = lineTop + lineHeight - 8;

  histogramCtx.strokeStyle = grid;
  for (let i = 1; i <= 3; i += 1) {
    const y = plotTop + (plotHeight * i) / 4;
    histogramCtx.beginPath();
    histogramCtx.moveTo(lineLeft + 6, y);
    histogramCtx.lineTo(lineLeft + lineWidth - 6, y);
    histogramCtx.stroke();
  }

  if (errorSeries.length > 0) {
    const maxError = Math.max(1, ...errorSeries);
    const seriesValues = recent.map((item) => {
      const value = Number(item.perception_error_seconds);
      return Number.isFinite(value) ? value : null;
    });

    histogramCtx.strokeStyle = "#d68a00";
    histogramCtx.lineWidth = 2;
    histogramCtx.beginPath();

    let started = false;
    seriesValues.forEach((value, idx) => {
      if (!Number.isFinite(value)) {
        return;
      }
      const x = lineLeft + 8 + (idx / Math.max(1, seriesValues.length - 1)) * (lineWidth - 16);
      const y = plotBottom - (value / maxError) * plotHeight;
      if (!started) {
        histogramCtx.moveTo(x, y);
        started = true;
      } else {
        histogramCtx.lineTo(x, y);
      }
    });

    if (started) {
      histogramCtx.stroke();
    }

    const latest = errorSeries[errorSeries.length - 1];
    const avgError = errorSeries.reduce((acc, val) => acc + val, 0) / errorSeries.length;
    histogramCtx.fillStyle = ink;
    histogramCtx.font = "11px Courier New";
    histogramCtx.fillText(`latest=${latest.toFixed(1)}s`, lineLeft + 8, metricsRowY);
    histogramCtx.fillText(`avg=${avgError.toFixed(1)}s`, lineLeft + Math.floor(lineWidth / 3), metricsRowY);
    histogramCtx.fillText(`max=${maxError.toFixed(1)}s`, lineLeft + Math.floor((lineWidth * 2) / 3), metricsRowY);
  } else {
    histogramCtx.fillStyle = ink;
    histogramCtx.font = "11px Courier New";
    histogramCtx.fillText("No error samples yet", lineLeft + 8, metricsRowY);
  }

  histogramCtx.fillStyle = ink;
  histogramCtx.font = "bold 12px Trebuchet MS";
  histogramCtx.fillText(
    `Fallback: ${fallbackCount}/${recent.length} successful frames | Realtime non-fatal: ${realtimeFailureCount}/${recent.length}`,
    16,
    h - 8
  );
}

function drawProcessingHistogram() {
  if (!histogramCtx || !histogramCanvas) {
    return;
  }

  if (histogramDrawRafId !== null) {
    return;
  }

  const schedule = typeof window !== "undefined" && typeof window.requestAnimationFrame === "function"
    ? window.requestAnimationFrame.bind(window)
    : (cb) => setTimeout(cb, 16);

  histogramDrawRafId = schedule(() => {
    histogramDrawRafId = null;
    if (histogramDrawInProgress) {
      return;
    }
    histogramDrawInProgress = true;
    try {
      drawProcessingHistogramNow();
    } catch (err) {
      // Prevent render exceptions from freezing future histogram updates.
      console.error("Histogram redraw failed:", err);
    } finally {
      histogramDrawInProgress = false;
    }
  });
}

function exportSymbolicLog() {
  const payload = JSON.stringify(symbolicHistory, null, 2);
  const blob = new Blob([payload], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");

  const now = new Date();
  const stamp = now.toISOString().replace(/[:.]/g, "-");
  anchor.href = url;
  anchor.download = `symbolic-log-${stamp}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

function drawTicks(radius) {
  if (!ctx) {
    return;
  }
  ctx.save();
  for (let i = 0; i < 60; i += 1) {
    const isHourTick = i % 5 === 0;
    const tickLength = isHourTick ? radius * 0.08 : radius * 0.04;
    const tickWidth = isHourTick ? 4 : 2;
    const angle = (i * Math.PI) / 30;

    const outerX = Math.sin(angle) * (radius - 12);
    const outerY = -Math.cos(angle) * (radius - 12);
    const innerX = Math.sin(angle) * (radius - 12 - tickLength);
    const innerY = -Math.cos(angle) * (radius - 12 - tickLength);

    ctx.beginPath();
    ctx.lineWidth = tickWidth;
    ctx.strokeStyle = currentClockTheme.tickColor;
    ctx.moveTo(innerX, innerY);
    ctx.lineTo(outerX, outerY);
    ctx.stroke();
  }
  ctx.restore();
}

function drawNumbers(radius) {
  if (!ctx) {
    return;
  }

  const roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];
  const numberStyle = currentClockTheme.numberStyle || "classic";
  const fontByStyle = {
    romanDeco: currentClockTheme.numberFont || "700 27px Georgia, serif",
    midCentury: currentClockTheme.numberFont || "700 27px Garamond, serif",
    quartz1980: "700 24px 'Lucida Console', Monaco, monospace",
    retro1970: "700 26px 'Arial Rounded MT Bold', 'Trebuchet MS', sans-serif",
    tech2000: "700 23px 'Courier New', monospace",
    classic: currentClockTheme.numberFont || "600 28px Georgia, serif",
  };

  ctx.save();
  ctx.fillStyle = currentClockTheme.numberColor;
  ctx.font = fontByStyle[numberStyle] || fontByStyle.classic;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (let num = 1; num <= 12; num += 1) {
    const angle = (num * Math.PI) / 6;
    const placementRadius = numberStyle === "tech2000" ? radius * 0.8 : radius * 0.78;
    const x = Math.sin(angle) * placementRadius;
    const y = -Math.cos(angle) * placementRadius;
    const label = numberStyle === "romanDeco" ? roman[num - 1] : String(num);

    if (numberStyle === "quartz1980") {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
      ctx.lineWidth = 1;
      ctx.strokeText(label, x, y);
    }

    if (numberStyle === "retro1970") {
      ctx.shadowColor = "rgba(115, 70, 34, 0.25)";
      ctx.shadowBlur = 4;
      ctx.shadowOffsetY = 1;
    } else {
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;
    }

    ctx.fillText(label, x, y);

    if (numberStyle === "romanDeco" && !artifactSafeMode) {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(16, 35, 58, 0.2)";
      ctx.lineWidth = 1;
      ctx.moveTo(x - 7, y + 12);
      ctx.lineTo(x + 7, y + 12);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawFacePattern(radius) {
  if (!ctx) {
    return;
  }

  const pattern = currentClockTheme.facePattern || "none";
  const allowDecorativeLines = !artifactSafeMode;

  ctx.save();
  ctx.beginPath();
  ctx.arc(0, 0, radius - 8, 0, Math.PI * 2);
  ctx.clip();

  if (pattern === "decoRings") {
    for (let r = radius * 0.3; r <= radius * 0.95; r += radius * 0.12) {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(25, 58, 93, 0.18)";
      ctx.lineWidth = 1.2;
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  if (pattern === "sunburst1950" && allowDecorativeLines) {
    for (let i = 0; i < 72; i += 1) {
      const a = (i * Math.PI * 2) / 72;
      const inner = radius * 0.2;
      const outer = radius * 0.95;
      ctx.beginPath();
      ctx.strokeStyle = i % 2 === 0 ? "rgba(120, 88, 42, 0.12)" : "rgba(255, 255, 255, 0.12)";
      ctx.lineWidth = 1;
      ctx.moveTo(Math.sin(a) * inner, -Math.cos(a) * inner);
      ctx.lineTo(Math.sin(a) * outer, -Math.cos(a) * outer);
      ctx.stroke();
    }
  }

  if (pattern === "grid1980" && allowDecorativeLines) {
    const step = radius * 0.16;
    ctx.strokeStyle = "rgba(50, 72, 128, 0.18)";
    ctx.lineWidth = 1;
    for (let p = -radius; p <= radius; p += step) {
      ctx.beginPath();
      ctx.moveTo(p, -radius);
      ctx.lineTo(p, radius);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(-radius, p);
      ctx.lineTo(radius, p);
      ctx.stroke();
    }
  }

  if (pattern === "target1970") {
    const rings = [0.95, 0.78, 0.61, 0.44, 0.27];
    const colors = [
      "rgba(168, 106, 49, 0.18)",
      "rgba(255, 241, 212, 0.22)",
      "rgba(122, 66, 88, 0.16)",
      "rgba(247, 216, 162, 0.22)",
      "rgba(79, 41, 62, 0.12)",
    ];
    rings.forEach((ratio, idx) => {
      ctx.beginPath();
      ctx.fillStyle = colors[idx];
      ctx.arc(0, 0, radius * ratio, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  if (pattern === "tech2000" && allowDecorativeLines) {
    for (let i = 0; i < 60; i += 1) {
      const a = (i * Math.PI) / 30;
      const r1 = radius * 0.83;
      const r2 = i % 5 === 0 ? radius * 0.9 : radius * 0.87;
      ctx.beginPath();
      ctx.strokeStyle = i % 5 === 0 ? "rgba(61, 104, 186, 0.25)" : "rgba(88, 133, 217, 0.14)";
      ctx.lineWidth = i % 5 === 0 ? 1.4 : 1;
      ctx.moveTo(Math.sin(a) * r1, -Math.cos(a) * r1);
      ctx.lineTo(Math.sin(a) * r2, -Math.cos(a) * r2);
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.strokeStyle = "rgba(48, 93, 178, 0.16)";
    ctx.lineWidth = 1;
    ctx.moveTo(-radius * 0.9, 0);
    ctx.lineTo(radius * 0.9, 0);
    ctx.moveTo(0, -radius * 0.9);
    ctx.lineTo(0, radius * 0.9);
    ctx.stroke();
  }

  ctx.restore();
}

function drawHand(angle, length, width, color) {
  if (!ctx) {
    return;
  }
  ctx.save();
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.lineCap = "round";
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.moveTo(0, rearY(15));
  ctx.lineTo(0, -length);
  ctx.stroke();
  ctx.restore();
}

function drawStyledHand(angle, length, width, color, style = "classic") {
  if (!ctx) {
    return;
  }

  ctx.save();
  ctx.rotate(angle);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;

  if (style === "spade1930") {
    ctx.beginPath();
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(2, width * 0.42);
    ctx.moveTo(0, rearY(15));
    ctx.lineTo(0, -length * 0.75);
    ctx.stroke();

    const tipY = -length * 0.84;
    const r = Math.max(4, width * 0.55);
    ctx.beginPath();
    ctx.arc(0, tipY, r, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.lineWidth = Math.max(2, width * 0.25);
    ctx.moveTo(0, tipY + r * 0.2);
    ctx.lineTo(0, tipY + r * 1.45);
    ctx.stroke();
    ctx.restore();
    return;
  }

  if (style === "dauphine1950") {
    const half = Math.max(2.5, width * 0.48);
    ctx.beginPath();
    ctx.moveTo(0, rearY(14));
    ctx.lineTo(half, -length * 0.18);
    ctx.lineTo(half * 0.65, -length * 0.92);
    ctx.lineTo(0, -length);
    ctx.lineTo(-half * 0.65, -length * 0.92);
    ctx.lineTo(-half, -length * 0.18);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    return;
  }

  if (style === "baton1980") {
    const half = Math.max(2.2, width * 0.55);
    ctx.beginPath();
    ctx.moveTo(-half, rearY(12));
    ctx.lineTo(half, rearY(12));
    ctx.lineTo(half, -length);
    ctx.lineTo(-half, -length);
    ctx.closePath();
    ctx.fill();

    ctx.globalAlpha = 0.22;
    ctx.fillRect(-half * 0.45, -length * 0.62, half * 0.9, length * 0.34);
    ctx.restore();
    return;
  }

  if (style === "block1970") {
    const half = Math.max(2.8, width * 0.62);
    const shoulderY = -length * 0.16;
    ctx.beginPath();
    ctx.moveTo(-half * 1.05, rearY(12));
    ctx.lineTo(half * 1.05, rearY(12));
    ctx.lineTo(half, shoulderY);
    ctx.lineTo(half * 0.72, -length);
    ctx.lineTo(-half * 0.72, -length);
    ctx.lineTo(-half, shoulderY);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    return;
  }

  if (style === "skeleton2000") {
    const half = Math.max(2.2, width * 0.55);
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(2, width * 0.34);
    ctx.beginPath();
    ctx.moveTo(0, rearY(15));
    ctx.lineTo(0, -length);
    ctx.stroke();

    ctx.beginPath();
    ctx.lineWidth = Math.max(2, width * 0.28);
    ctx.arc(0, -length * 0.53, half * 1.2, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    return;
  }

  if (style === "lollipop1980") {
    ctx.beginPath();
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(2, width * 0.9);
    ctx.moveTo(0, rearY(18));
    ctx.lineTo(0, -length);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(0, -length * 0.68, Math.max(3, width * 1.15), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return;
  }

  if (style === "rail1930") {
    ctx.beginPath();
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(1.5, width * 0.8);
    ctx.moveTo(0, rearY(18));
    ctx.lineTo(0, -length);
    ctx.stroke();

    if (!artifactSafeMode) {
      ctx.beginPath();
      ctx.lineWidth = Math.max(1.2, width * 0.42);
      ctx.moveTo(-Math.max(5, width * 1.45), -length * 0.86);
      ctx.lineTo(Math.max(5, width * 1.45), -length * 0.86);
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  if (style === "needle1950") {
    ctx.beginPath();
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(1.6, width * 0.74);
    ctx.moveTo(0, rearY(18));
    ctx.lineTo(0, -length);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, -length);
    ctx.lineTo(Math.max(2, width * 0.55), -length * 0.88);
    ctx.lineTo(-Math.max(2, width * 0.55), -length * 0.88);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    return;
  }

  drawHand(0, length, width, color);
  ctx.restore();
}

function getRandomClockTheme() {
  if (CLOCK_THEMES.length <= 1) {
    return CLOCK_THEMES[0];
  }

  let selected = currentClockTheme;
  while (selected === currentClockTheme) {
    const idx = Math.floor(Math.random() * CLOCK_THEMES.length);
    selected = CLOCK_THEMES[idx];
  }
  return selected;
}

function randomizeClockTheme() {
  currentClockTheme = getRandomClockTheme();
  drawClock();
  syncThemeSelectValue();
  persistClockFaceState();

  if (symbolicOutput) {
    symbolicOutput.value = `Applied random clock theme: ${currentClockTheme.name} (${currentClockTheme.era})`;
    resizeSymbolicOutput();
  }
}

function resetClockFace(silent = false) {
  currentClockTheme = DEFAULT_PLAIN_CLOCK_THEME;
  isFlipped = false;
  clockRotationDegrees = 0;
  clock3DRotationDegrees = 0;

  randomizeThemeEverySecond = false;
  rotateClockEverySecond = false;
  randomFlipEverySecond = false;
  rotateClock3DEverySecond = false;
  showExtraDistractorLine = false;
  artifactSafeMode = true;

  localStorage.setItem(RANDOMIZE_EVERY_SECOND_STORAGE_KEY, "false");
  localStorage.setItem(ROTATION_INCREMENT_STORAGE_KEY, "false");
  localStorage.setItem(RANDOM_FLIP_STORAGE_KEY, "false");
  localStorage.setItem(ROTATE_3D_STORAGE_KEY, "false");
  localStorage.setItem(EXTRA_LINE_STORAGE_KEY, "false");
  localStorage.setItem(ARTIFACT_SAFE_MODE_STORAGE_KEY, "true");

  if (randomizeEverySecondToggle) {
    randomizeEverySecondToggle.checked = false;
  }
  if (rotationIncrementToggle) {
    rotationIncrementToggle.checked = false;
  }
  if (randomFlipToggle) {
    randomFlipToggle.checked = false;
  }
  if (rotate3DToggle) {
    rotate3DToggle.checked = false;
  }
  if (extraLineToggle) {
    extraLineToggle.checked = false;
  }
  if (artifactSafeModeToggle) {
    artifactSafeModeToggle.checked = true;
  }

  drawClock();
  syncThemeSelectValue();
  persistClockFaceState();

  if (!silent && symbolicOutput) {
    symbolicOutput.value = "Clock face reset to original plain style, orientation, and motion toggles.";
    resizeSymbolicOutput();
  }
}

function drawClock() {
  if (!ctx || !canvas) {
    return;
  }

  const now = new Date();
  const hour = now.getHours() % 12;
  const minute = now.getMinutes();
  const second = now.getSeconds();

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const radius = Math.min(canvas.width, canvas.height) * 0.45;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  if (isFlipped) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }

  ctx.translate(cx, cy);

  const threeDRotationRadians = (clock3DRotationDegrees * Math.PI) / 180;
  const threeDCosine = Math.cos(threeDRotationRadians);
  const threeDScaleX = Math.max(0.12, Math.abs(threeDCosine));
  const isBackView = threeDCosine < 0;

  ctx.scale(threeDScaleX, 1);
  if (isBackView) {
    // Mirror while on the rear side so the dial appears from behind.
    ctx.scale(-1, 1);
  }

  if (clockRotationDegrees !== 0) {
    ctx.rotate((clockRotationDegrees * Math.PI) / 180);
  }

  const faceGradient = ctx.createRadialGradient(0, 0, radius * 0.15, 0, 0, radius);
  faceGradient.addColorStop(0, currentClockTheme.faceInnerColor || "#ffffff");
  faceGradient.addColorStop(1, currentClockTheme.faceOuterColor || "#d9ebff");

  ctx.beginPath();
  ctx.fillStyle = faceGradient;
  ctx.arc(0, 0, radius, 0, Math.PI * 2);
  ctx.fill();

  drawFacePattern(radius);

  ctx.lineWidth = 10;
  ctx.strokeStyle = currentClockTheme.bezelColor || "#1f5f8b";
  ctx.stroke();

  drawTicks(radius);
  drawNumbers(radius);
  drawFaceDistractorLine(radius);

  const minuteAngle = ((minute + second / 60) / 60) * Math.PI * 2;
  const hourAngle = ((hour + minute / 60 + second / 3600) / 12) * Math.PI * 2;
  const secondAngle = (second / 60) * Math.PI * 2;

  drawStyledHand(
    hourAngle,
    radius * 0.48,
    10,
    currentClockTheme.hourColor,
    currentClockTheme.hourHandStyle,
  );
  drawStyledHand(
    minuteAngle,
    radius * 0.72,
    7,
    currentClockTheme.minuteColor,
    currentClockTheme.minuteHandStyle,
  );
  drawStyledHand(
    secondAngle,
    radius * 0.78,
    3,
    currentClockTheme.secondColor,
    currentClockTheme.secondHandStyle,
  );

  ctx.beginPath();
  ctx.fillStyle = currentClockTheme.centerOuterColor;
  ctx.arc(0, 0, 8, 0, Math.PI * 2);
  ctx.fill();

  ctx.beginPath();
  ctx.fillStyle = currentClockTheme.centerInnerColor;
  ctx.arc(0, 0, 3, 0, Math.PI * 2);
  ctx.fill();

  if (isBackView) {
    // Subtle darkening helps communicate we are looking at the back side.
    ctx.beginPath();
    ctx.globalAlpha = 0.16;
    ctx.fillStyle = "#0f1726";
    ctx.arc(0, 0, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  ctx.restore();
}

async function captureSnapshot() {
  if (!canvas) {
    throw new Error("Canvas not available");
  }

  const imageData = canvas.toDataURL("image/png");
  lastCaptureAt = new Date();

  const response = await fetch("/save-image", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      imageData,
      flipped: isFlipped,
    }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Failed to save snapshot");
  }

  if (symbolicOutput) {
    if (realtimeTimer === null) {
      symbolicOutput.value = `Snapshot saved at ${lastCaptureAt.toLocaleTimeString()}\nflipped=${String(data.flipped)}`;
      resizeSymbolicOutput();
    }
  }
}

function buildProcessedImageUrl(savedImageUrl) {
  if (!savedImageUrl) {
    return null;
  }

  const cacheBust = Date.now();
  const separator = savedImageUrl.includes("?") ? "&" : "?";
  return `${savedImageUrl}${separator}t=${cacheBust}`;
}

function buildProcessedStagesJsonUrl(processedJsonUrl) {
  if (!processedJsonUrl) {
    return null;
  }

  const cacheBust = Date.now();
  const separator = processedJsonUrl.includes("?") ? "&" : "?";
  return `${processedJsonUrl}${separator}t=${cacheBust}`;
}

function preloadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Image preload failed"));
    image.src = url;
  });
}

function renderJsonToPre(element, value, fallbackText) {
  if (!element) {
    return;
  }
  if (value === null || typeof value === "undefined") {
    element.textContent = fallbackText;
    return;
  }
  element.textContent = JSON.stringify(value, null, 2);
}

function ensureProcessedImagePreviewElement() {
  if (!processedImagePreview) {
    return null;
  }

  if (!processedPreviewImgEl) {
    const existing = processedImagePreview.querySelector("img");
    processedPreviewImgEl = existing || document.createElement("img");
    processedPreviewImgEl.alt = "Processed clock preview";
  }

  return processedPreviewImgEl;
}

function ensureOpenCvStagesStructure() {
  if (!opencvStagesList || opencvStagesInitialized) {
    return;
  }

  opencvStagesList.innerHTML = "";

  OPENCV_STAGE_ORDER.forEach((key) => {
    const card = document.createElement("div");
    card.className = "opencv-stage-card";

    const title = document.createElement("strong");
    title.textContent = OPENCV_STAGE_LABELS[key] || key;

    const content = document.createElement("div");
    content.className = "opencv-stage-content";

    const imageWrap = document.createElement("div");
    imageWrap.className = "opencv-stage-image-wrap";

    const img = document.createElement("img");
    img.alt = `${title.textContent} stage`;

    imageWrap.appendChild(img);

    const dataWrap = document.createElement("div");
    dataWrap.className = "opencv-stage-data";

    const details = document.createElement("details");
    details.className = "opencv-stage-data-details";

    const summary = document.createElement("summary");
    summary.textContent = "Show stage data, vectors, masks, and metadata";

    const metaLabel = document.createElement("p");
    metaLabel.className = "opencv-stage-data-label";
    metaLabel.textContent = "Stage metadata";

    const metaPre = document.createElement("pre");
    metaPre.className = "opencv-stage-array";

    const dataLabel = document.createElement("p");
    dataLabel.className = "opencv-stage-data-label";
    dataLabel.textContent = "Numeric data array";

    const dataPre = document.createElement("pre");
    dataPre.className = "opencv-stage-array";

    const masksLabel = document.createElement("p");
    masksLabel.className = "opencv-stage-data-label";
    masksLabel.textContent = "Masks";

    const masksPre = document.createElement("pre");
    masksPre.className = "opencv-stage-array";

    const vectorsLabel = document.createElement("p");
    vectorsLabel.className = "opencv-stage-data-label";
    vectorsLabel.textContent = "Symbolic vectors";

    const vectorsPre = document.createElement("pre");
    vectorsPre.className = "opencv-stage-array";

    const debugLabel = document.createElement("p");
    debugLabel.className = "opencv-stage-data-label";
    debugLabel.textContent = "Debug metadata";

    const debugPre = document.createElement("pre");
    debugPre.className = "opencv-stage-array";

    details.appendChild(summary);
    details.appendChild(metaLabel);
    details.appendChild(metaPre);
    details.appendChild(dataLabel);
    details.appendChild(dataPre);
    details.appendChild(masksLabel);
    details.appendChild(masksPre);
    details.appendChild(vectorsLabel);
    details.appendChild(vectorsPre);
    details.appendChild(debugLabel);
    details.appendChild(debugPre);
    dataWrap.appendChild(details);

    content.appendChild(imageWrap);
    content.appendChild(dataWrap);

    card.appendChild(title);
    card.appendChild(content);
    opencvStagesList.appendChild(card);

    const stageView = {
      cardElement: card,
      imageElement: img,
      dataDetails: details,
      metaPre,
      dataPre,
      masksPre,
      vectorsPre,
      debugPre,
      stageData: null,
      stageMeta: null,
      masks: null,
      symbolicVectors: null,
      debugMetadata: null,
    };
    opencvStageElements.set(key, stageView);

    details.addEventListener("toggle", () => {
      if (!details.open) {
        metaPre.textContent = "";
        dataPre.textContent = "";
        masksPre.textContent = "";
        vectorsPre.textContent = "";
        debugPre.textContent = "";
        return;
      }

      renderJsonToPre(metaPre, stageView.stageMeta, "No stage metadata available.");
      renderJsonToPre(dataPre, stageView.stageData, "No numeric data array available.");
      renderJsonToPre(masksPre, stageView.masks, "No mask data available.");
      renderJsonToPre(vectorsPre, stageView.symbolicVectors, "No symbolic vectors available.");
      renderJsonToPre(debugPre, stageView.debugMetadata, "No debug metadata available.");
    });
  });

  opencvStagesInitialized = true;
  updateOpenCvStageVisibility();
}

function normalizeOpenCvStageKey(stageKey) {
  if (typeof stageKey === "string" && OPENCV_STAGE_ORDER.includes(stageKey)) {
    return stageKey;
  }
  return DEFAULT_OPENCV_STAGE_KEY;
}

function updateOpenCvStageVisibility() {
  const selectedKey = normalizeOpenCvStageKey(selectedOpenCvStageKey);
  selectedOpenCvStageKey = selectedKey;

  if (opencvStageSelect && opencvStageSelect.value !== selectedKey) {
    opencvStageSelect.value = selectedKey;
  }

  opencvStageElements.forEach((stageView, key) => {
    if (!stageView || !stageView.cardElement) {
      return;
    }
    stageView.cardElement.hidden = key !== selectedKey;
  });
}

function setProcessedImagePreview(savedImageUrl, { preserveExistingWhenMissing = false, updateSeq = null } = {}) {
  if (!processedImagePreview) {
    return;
  }

  const seq = Number.isFinite(Number(updateSeq)) ? Number(updateSeq) : (processedPreviewUpdateSeq + 1);
  processedPreviewUpdateSeq = Math.max(processedPreviewUpdateSeq, seq);

  if (savedImageUrl) {
    const previewUrl = buildProcessedImageUrl(savedImageUrl);
    const imgEl = ensureProcessedImagePreviewElement();
    if (!imgEl) {
      return;
    }

    preloadImage(previewUrl)
      .then(() => {
        if (seq !== processedPreviewUpdateSeq) {
          return;
        }
        if (!processedImagePreview.contains(imgEl)) {
          processedImagePreview.innerHTML = "";
          processedImagePreview.appendChild(imgEl);
        }
        if (imgEl.src !== previewUrl) {
          imgEl.src = previewUrl;
        }
      })
      .catch(() => {
        // Keep the previously rendered image if preload fails to avoid flashing placeholders.
      });

    if (processedImageDownloadLink) {
      processedImageDownloadLink.href = previewUrl;
      processedImageDownloadLink.download = "processed_clock.png";
      processedImageDownloadLink.hidden = false;
      processedImageDownloadLink.textContent = "Download Processed Image";
    }
  } else {
    if (preserveExistingWhenMissing) {
      return;
    }
    processedPreviewImgEl = null;
    processedImagePreview.innerHTML = '<span class="processed-image-placeholder">No processed image yet. Convert to Symbolic to save one.</span>';
    if (processedImageDownloadLink) {
      processedImageDownloadLink.hidden = true;
      processedImageDownloadLink.removeAttribute("href");
      processedImageDownloadLink.removeAttribute("download");
      processedImageDownloadLink.textContent = "Download Processed Image";
    }
  }
}

function setOpenCvStagesPreview(stagesPayload) {
  if (!opencvStagesList) {
    return;
  }

  if (!stagesPayload || typeof stagesPayload !== "object") {
    latestOpenCvStagesPayload = null;
    if (!opencvStagesInitialized) {
      opencvStagesList.innerHTML = '<span class="processed-image-placeholder">No OpenCV stages yet. Convert to Symbolic to generate them.</span>';
    }
    return;
  }

  latestOpenCvStagesPayload = stagesPayload;
  ensureOpenCvStagesStructure();
  updateOpenCvStageVisibility();

  const masksPayload = stagesPayload.masks && typeof stagesPayload.masks === "object" ? stagesPayload.masks : null;
  const vectorsPayload = stagesPayload.symbolic_vectors && typeof stagesPayload.symbolic_vectors === "object"
    ? stagesPayload.symbolic_vectors
    : null;
  const debugPayload = stagesPayload.debug && typeof stagesPayload.debug === "object" ? stagesPayload.debug : null;
  const selectedStageKey = normalizeOpenCvStageKey(selectedOpenCvStageKey);

  const stageEntry = stagesPayload[selectedStageKey];
  const selectedStageView = opencvStageElements.get(selectedStageKey);
  if (!selectedStageView) {
    return;
  }

  let base64Value = null;
  let stageData = null;
  let stageMeta = null;

  if (typeof stageEntry === "string") {
    base64Value = stageEntry;
  } else if (stageEntry && typeof stageEntry === "object") {
    if (typeof stageEntry.image === "string") {
      base64Value = stageEntry.image;
    }
    if (Object.prototype.hasOwnProperty.call(stageEntry, "data")) {
      stageData = stageEntry.data;
    }

    stageMeta = {
      shape: stageEntry.shape ?? null,
      dtype: stageEntry.dtype ?? null,
      min: stageEntry.min ?? null,
      max: stageEntry.max ?? null,
    };
  }

  if (typeof base64Value === "string" && base64Value.trim()) {
    const nextSrc = `data:image/png;base64,${base64Value}`;
    if (selectedStageView.imageElement.src !== nextSrc) {
      selectedStageView.imageElement.src = nextSrc;
    }
  }

  selectedStageView.stageData = stageData;
  selectedStageView.stageMeta = stageMeta;
  selectedStageView.masks = masksPayload;
  selectedStageView.symbolicVectors = vectorsPayload;
  selectedStageView.debugMetadata = debugPayload;

  if (selectedStageView.dataDetails.open) {
    renderJsonToPre(selectedStageView.metaPre, selectedStageView.stageMeta, "No stage metadata available.");
    renderJsonToPre(selectedStageView.dataPre, selectedStageView.stageData, "No numeric data array available.");
    renderJsonToPre(selectedStageView.masksPre, selectedStageView.masks, "No mask data available.");
    renderJsonToPre(selectedStageView.vectorsPre, selectedStageView.symbolicVectors, "No symbolic vectors available.");
    renderJsonToPre(selectedStageView.debugPre, selectedStageView.debugMetadata, "No debug metadata available.");
  } else {
    selectedStageView.metaPre.textContent = "";
    selectedStageView.dataPre.textContent = "";
    selectedStageView.masksPre.textContent = "";
    selectedStageView.vectorsPre.textContent = "";
    selectedStageView.debugPre.textContent = "";
  }
}

function setOpenCvStagesDownloadLink(processedDataUrl) {
  if (!opencvStagesDownloadLink) {
    return;
  }

  if (processedDataUrl) {
    const downloadUrl = buildProcessedStagesJsonUrl(processedDataUrl);
    opencvStagesDownloadLink.href = downloadUrl;
    opencvStagesDownloadLink.download = "processed_clock_data.json";
    opencvStagesDownloadLink.hidden = false;
    opencvStagesDownloadLink.textContent = "Download OpenCV Processing Stages";
    return;
  }

  opencvStagesDownloadLink.hidden = true;
  opencvStagesDownloadLink.removeAttribute("href");
  opencvStagesDownloadLink.removeAttribute("download");
  opencvStagesDownloadLink.textContent = "Download OpenCV Processing Stages";
}

function resetProcessedArtifactsPanels() {
  setProcessedImagePreview(null);
  setOpenCvStagesDownloadLink(null);
  latestOpenCvStagesPayload = null;
  opencvStagesInitialized = false;
  opencvStageElements.clear();
  selectedOpenCvStageKey = DEFAULT_OPENCV_STAGE_KEY;

  if (opencvStageSelect) {
    opencvStageSelect.value = DEFAULT_OPENCV_STAGE_KEY;
  }

  if (opencvStagesList) {
    opencvStagesList.innerHTML = '<span class="processed-image-placeholder">No OpenCV stages yet. Convert to Symbolic to generate them.</span>';
  }
}

async function loadOpenCvStages(processedDataUrl, { preserveExistingWhenMissing = false, updateSeq = null } = {}) {
  const seq = Number.isFinite(Number(updateSeq)) ? Number(updateSeq) : (openCvStagesUpdateSeq + 1);
  openCvStagesUpdateSeq = Math.max(openCvStagesUpdateSeq, seq);

  if (!processedDataUrl) {
    if (preserveExistingWhenMissing) {
      return;
    }
    setOpenCvStagesDownloadLink(null);
    setOpenCvStagesPreview(null);
    return;
  }

  try {
    const jsonUrl = buildProcessedStagesJsonUrl(processedDataUrl);
    const response = await fetch(jsonUrl);
    if (!response.ok) {
      throw new Error("Unable to load OpenCV processing stages");
    }
    const payload = await response.json();
    if (seq !== openCvStagesUpdateSeq) {
      return;
    }
    setOpenCvStagesPreview(payload);
    setOpenCvStagesDownloadLink(jsonUrl);
  } catch (err) {
    // Preserve currently displayed stages if one refresh fails.
  }
}

async function convertToSymbolic() {
  const isRealtime = realtimeTimer !== null;
  const shouldRefreshProcessedPreview = !isRealtime || (realtimeFrameCount % REALTIME_PREVIEW_REFRESH_EVERY === 0);
  const shouldRefreshProcessingStages = realtimeTimer === null || (realtimeFrameCount % REALTIME_STAGE_REFRESH_EVERY === 0);
  const imageData = canvas ? canvas.toDataURL("image/png") : "";
  const response = await fetch("/vision/clock", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      image: imageData,
      save_image: true,
      flipped: isFlipped,
      realtime_mode: isRealtime,
      include_processing_stages: shouldRefreshProcessingStages,
      hand_colors: {
        hour: currentClockTheme && typeof currentClockTheme.hourColor === "string" ? currentClockTheme.hourColor : null,
        minute: currentClockTheme && typeof currentClockTheme.minuteColor === "string" ? currentClockTheme.minuteColor : null,
        second: currentClockTheme && typeof currentClockTheme.secondColor === "string" ? currentClockTheme.secondColor : null,
      },
    }),
  });

  const data = await response.json();
  if (!response.ok) {
    const err = new Error(data.error || "Failed symbolic conversion");
    err.payload = data;
    err.responseStatus = response.status;
    if (isRealtime) {
      appendSymbolicFailureLog(String(err.message || "Realtime symbolic conversion failed"));
      appendProcessingMetrics(data, { fromRealtimeFailure: true });
      if (symbolicOutput) {
        symbolicOutput.value = `Realtime symbolic conversion failed: ${String(err.message || "unknown error")}`;
        resizeSymbolicOutput();
      }
      return;
    }
    throw err;
  }

  const symbolic = data.symbolic || data;
  const responseMetrics = extractDebugMetricsFromPayload(data);
  if (responseMetrics && symbolic && typeof symbolic === "object" && !symbolic.debug_metrics) {
    symbolic.debug_metrics = responseMetrics;
  }
  const captureNote = lastCaptureAt
    ? `Based on snapshot captured at ${lastCaptureAt.toLocaleTimeString()}\n\n`
    : "Based on latest saved snapshot\n\n";

  if (symbolicOutput) {
    symbolicOutput.value = captureNote + JSON.stringify(symbolic, null, 2);
    resizeSymbolicOutput();
  }

  if (probabilisticReasoningOutput) {
    probabilisticReasoningOutput.value = formatProbabilisticReasoning(symbolic.probabilistic_reasoning);
    resizeProbabilisticReasoningOutput();
  }

  if (shouldRefreshProcessedPreview) {
    setProcessedImagePreview(data.saved_image_url || null, {
      preserveExistingWhenMissing: isRealtime,
      updateSeq: processedPreviewUpdateSeq + 1,
    });
  }

  if (shouldRefreshProcessingStages) {
    const stageLoadPromise = loadOpenCvStages(data.processed_data_url || data.processed_json_url || null, {
      preserveExistingWhenMissing: isRealtime,
      updateSeq: openCvStagesUpdateSeq + 1,
    });
    if (!isRealtime) {
      await stageLoadPromise;
    }
  }
  appendSymbolicLog(symbolic);
  appendProcessingMetrics(symbolic);
}

async function captureAndConvertOnce() {
  if (realtimeBusy) {
    return;
  }

  realtimeBusy = true;
  if (realtimeTimer !== null) {
    realtimeFrameCount += 1;
  }
  try {
    await captureSnapshot();
    try {
      await convertToSymbolic();
    } catch (err) {
      if (isNonFatalRealtimeError(err)) {
        const errorMessage = String(err && err.message ? err.message : err);
        appendSymbolicFailureLog(errorMessage);
        const failureMetrics = err && err.payload ? err.payload : null;
        appendProcessingMetrics(failureMetrics, { fromRealtimeFailure: true });
        if (symbolicOutput) {
          symbolicOutput.value = `Realtime non-fatal conversion error recorded: ${errorMessage}`;
          resizeSymbolicOutput();
        }
        return;
      }
      throw err;
    }
    updateRealtimeLimitInfo();
  } finally {
    realtimeBusy = false;
  }
}

function setRealtimeUiState(isActive) {
  if (realtimeBtn) {
    realtimeBtn.textContent = isActive
      ? "Stop Realtime Capture + Process"
      : "Start Realtime Capture + Process";
    realtimeBtn.classList.toggle("realtime-active", isActive);
  }

  const locked = Boolean(isActive);
  if (clearDataBtn) {
    clearDataBtn.disabled = locked;
  }
  if (clearWorkspaceBtn) {
    clearWorkspaceBtn.disabled = locked;
  }
  if (exportLogBtn) {
    exportLogBtn.disabled = locked;
  }
  if (openReadmeBtn) {
    openReadmeBtn.disabled = locked;
  }
}

function handleRealtimeFatalError(err) {
  if (symbolicOutput) {
    symbolicOutput.value = String(err && err.message ? err.message : err);
    resizeSymbolicOutput();
  }
  stopRealtimeMode();
}

function stopRealtimeMode() {
  if (realtimeTimer !== null) {
    clearInterval(realtimeTimer);
    realtimeTimer = null;
  }
  if (realtimeMaxStopTimer !== null) {
    clearTimeout(realtimeMaxStopTimer);
    realtimeMaxStopTimer = null;
  }
  realtimeRunStartedAt = null;
  realtimeFrameCount = 0;
  setRealtimeUiState(false);
  updateRealtimeLimitInfo();
}

function resetRealtimeRunHistory() {
  symbolicHistory.length = 0;
  processingHistory.length = 0;
  lastKnownDebugMetrics = null;

  if (symbolicLogOutput) {
    symbolicLogOutput.value = "";
    symbolicLogOutput.scrollTop = 0;
  }

  if (symbolicOutput) {
    symbolicOutput.value = "";
    resizeSymbolicOutput();
  }

  if (probabilisticReasoningOutput) {
    probabilisticReasoningOutput.value = "";
    resizeProbabilisticReasoningOutput();
  }

  drawProcessingHistogram();
  persistAcquiredData();
}

function startRealtimeMode() {
  if (realtimeTimer !== null) {
    return;
  }

  // Start each realtime session from a clean log/history state to avoid catch-up.
  resetRealtimeRunHistory();

  setRealtimeUiState(true);
  realtimeRunStartedAt = Date.now();
  updateRealtimeLimitInfo();

  captureAndConvertOnce().catch(handleRealtimeFatalError);

  realtimeTimer = setInterval(() => {
    captureAndConvertOnce().catch(handleRealtimeFatalError);
  }, REALTIME_INTERVAL_MS);

  if (REALTIME_MAX_SECONDS !== null) {
    realtimeMaxStopTimer = setTimeout(() => {
      stopRealtimeMode();
      if (symbolicOutput) {
        symbolicOutput.value = `Realtime capture stopped after reaching the configured max duration (${REALTIME_MAX_SECONDS}s).`;
        resizeSymbolicOutput();
      }
    }, REALTIME_MAX_SECONDS * 1000);
  }
}

if (flipBtn) {
  flipBtn.addEventListener("click", () => {
    isFlipped = !isFlipped;
    drawClock();
    persistClockFaceState();
  });
}

if (resetClockBtn) {
  resetClockBtn.addEventListener("click", () => {
    resetClockFace();
  });
}

if (randomThemeBtn) {
  randomThemeBtn.addEventListener("click", () => {
    randomizeClockTheme();
  });
}

if (themeSelect) {
  themeSelect.addEventListener("change", () => {
    const selectedTheme = getThemeByName(themeSelect.value);
    if (!selectedTheme) {
      return;
    }

    currentClockTheme = selectedTheme;
    drawClock();
    persistClockFaceState();

    if (symbolicOutput) {
      symbolicOutput.value = `Applied selected theme: ${currentClockTheme.name} (${currentClockTheme.era}).`;
      resizeSymbolicOutput();
    }
  });
}

if (randomizeEverySecondToggle) {
  randomizeEverySecondToggle.addEventListener("change", () => {
    randomizeThemeEverySecond = randomizeEverySecondToggle.checked;
    localStorage.setItem(RANDOMIZE_EVERY_SECOND_STORAGE_KEY, String(randomizeThemeEverySecond));
    persistClockFaceState();
  });
}

if (rotationIncrementToggle) {
  rotationIncrementToggle.addEventListener("change", () => {
    rotateClockEverySecond = rotationIncrementToggle.checked;
    localStorage.setItem(ROTATION_INCREMENT_STORAGE_KEY, String(rotateClockEverySecond));
    persistClockFaceState();
  });
}

if (randomFlipToggle) {
  randomFlipToggle.addEventListener("change", () => {
    randomFlipEverySecond = randomFlipToggle.checked;
    localStorage.setItem(RANDOM_FLIP_STORAGE_KEY, String(randomFlipEverySecond));
    persistClockFaceState();
  });
}

if (rotate3DToggle) {
  rotate3DToggle.addEventListener("change", () => {
    rotateClock3DEverySecond = rotate3DToggle.checked;
    localStorage.setItem(ROTATE_3D_STORAGE_KEY, String(rotateClock3DEverySecond));
    persistClockFaceState();
  });
}

if (extraLineToggle) {
  extraLineToggle.addEventListener("change", () => {
    showExtraDistractorLine = extraLineToggle.checked;
    localStorage.setItem(EXTRA_LINE_STORAGE_KEY, String(showExtraDistractorLine));
    drawClock();
    persistClockFaceState();
  });
}

if (artifactSafeModeToggle) {
  artifactSafeModeToggle.addEventListener("change", () => {
    artifactSafeMode = artifactSafeModeToggle.checked;
    localStorage.setItem(ARTIFACT_SAFE_MODE_STORAGE_KEY, String(artifactSafeMode));
    drawClock();
    persistClockFaceState();
  });
}

if (opencvStageSelect) {
  opencvStageSelect.addEventListener("change", () => {
    selectedOpenCvStageKey = normalizeOpenCvStageKey(opencvStageSelect.value);
    updateOpenCvStageVisibility();
    if (latestOpenCvStagesPayload) {
      setOpenCvStagesPreview(latestOpenCvStagesPayload);
    }
  });
}

if (addAllGroundingPresetsBtn) {
  addAllGroundingPresetsBtn.addEventListener("click", () => {
    addAllGroundingPresetsToSequence();
  });
}

if (applyGroundingScriptBtn) {
  applyGroundingScriptBtn.addEventListener("click", () => {
    applyGroundingScriptSequence();
  });
}

if (copyGroundingScriptBtn) {
  copyGroundingScriptBtn.addEventListener("click", async () => {
    await copyGroundingScriptToClipboard();
  });
}

if (clearGroundingScriptBtn) {
  clearGroundingScriptBtn.addEventListener("click", () => {
    clearGroundingScriptSequence();
  });
}

if (symbolicBtn) {
  symbolicBtn.addEventListener("click", async () => {
    try {
      await captureAndConvertOnce();
      persistAcquiredData();
    } catch (err) {
      if (symbolicOutput) {
        symbolicOutput.value = String(err.message || err);
        resizeSymbolicOutput();
        persistAcquiredData();
      }
    }
  });
}

if (realtimeBtn) {
  realtimeBtn.addEventListener("click", async () => {
    if (realtimeTimer !== null) {
      stopRealtimeMode();
      return;
    }

    if (isPasswordPromptRequired("start_realtime_capture")) {
      const password = await promisePassword("start realtime capture + process");
      if (!password) {
        if (symbolicOutput) {
          symbolicOutput.value = "Realtime capture cancelled.";
          resizeSymbolicOutput();
        }
        return;
      }

      const isValid = await verifyPassword(password);
      if (!isValid) {
        if (symbolicOutput) {
          symbolicOutput.value = "Invalid password. Realtime capture cancelled.";
          resizeSymbolicOutput();
        }
        return;
      }
    }

    startRealtimeMode();
  });
}

if (exportLogBtn) {
  exportLogBtn.addEventListener("click", () => {
    if (symbolicHistory.length === 0) {
      if (symbolicOutput) {
        symbolicOutput.value = "No symbolic outputs logged yet.";
        resizeSymbolicOutput();
      }
      return;
    }
    exportSymbolicLog();
  });
}

if (openReadmeBtn) {
  openReadmeBtn.addEventListener("click", () => {
    window.open("/project-readme", "_blank", "noopener,noreferrer");
  });
}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", () => {
    toggleTheme();
  });
}

if (clearDataBtn) {
  clearDataBtn.addEventListener("click", async () => {
    if (isPasswordPromptRequired("clear_data")) {
      const password = await promisePassword("clear all data");
      if (!password) {
        updateExperimentsStatus("Clear data cancelled.");
        return;
      }

      const isValid = await verifyPassword(password);
      if (!isValid) {
        updateExperimentsStatus("Invalid password. Clear data cancelled.");
        return;
      }
    }

    const confirmed = window.confirm(
      "Clear all data and workspace state? This will erase symbolic output, logs, histogram stats, grounding script presets, Processed Image, OpenCV Processing Stages, and reset the clock face."
    );
    if (!confirmed) {
      return;
    }
    try {
      await clearDataAndStats();
    } catch (err) {
      updateExperimentsStatus(String(err && err.message ? err.message : err));
    }
  });
}

if (clearWorkspaceBtn) {
  clearWorkspaceBtn.addEventListener("click", async () => {
    if (isPasswordPromptRequired("clear_workspace")) {
      const password = await promisePassword("clear workspace");
      if (!password) {
        updateExperimentsStatus("Clear workspace cancelled.");
        return;
      }

      const isValid = await verifyPassword(password);
      if (!isValid) {
        updateExperimentsStatus("Invalid password. Clear workspace cancelled.");
        return;
      }
    }

    const confirmed = window.confirm(
      "Clear workspace data? This resets the clock face to defaults and clears symbolic output, log history, processing histogram data, Processed Image, and OpenCV Processing Stages. Reproducible Experiments (Run Sweep) and LLM Benchmark Comparison results are preserved."
    );
    if (!confirmed) {
      return;
    }
    try {
      await clearWorkspaceData();
    } catch (err) {
      updateExperimentsStatus(String(err && err.message ? err.message : err));
    }
  });
}

if (runExperimentsBtn) {
  runExperimentsBtn.addEventListener("click", async () => {
    if (experimentsBusy) {
      return;
    }
    await runExperimentSweepFromUI();
  });
}

if (clearExperimentsBtn) {
  clearExperimentsBtn.addEventListener("click", async () => {
    if (isPasswordPromptRequiredWithFallback("clear_experiments", "run_experiments")) {
      const password = await promisePassword("clear sweep results");
      if (!password) {
        updateExperimentsStatus("Sweep clear cancelled.");
        return;
      }

      const isValid = await verifyPassword(password);
      if (!isValid) {
        updateExperimentsStatus("Invalid password. Sweep clear cancelled.");
        return;
      }
    }

    const confirmed = window.confirm(
      "Clear the current sweep summary, progress, thumbnails, and generated experiment artifacts?"
    );
    if (!confirmed) {
      return;
    }

    try {
      await clearExperimentSectionData();
      updateExperimentsStatus("Experiment sweep results cleared.");
    } catch (err) {
      updateExperimentsStatus(String(err && err.message ? err.message : err));
    }
  });
}

if (refreshExperimentsBtn) {
  refreshExperimentsBtn.addEventListener("click", async () => {
    if (experimentsBusy) {
      return;
    }
    await refreshExperimentsSummary();
  });
}

if (downloadConditionMatrixBtn) {
  downloadConditionMatrixBtn.addEventListener("click", () => {
    downloadExperimentArtifact("condition_matrix.csv");
  });
}

if (downloadMetricsBtn) {
  downloadMetricsBtn.addEventListener("click", () => {
    downloadExperimentArtifact("metrics_by_condition.csv");
  });
}

if (downloadStatsCsvBtn) {
  downloadStatsCsvBtn.addEventListener("click", () => {
    downloadExperimentArtifact("stats_by_condition.csv");
  });
}

if (downloadStatsSummaryBtn) {
  downloadStatsSummaryBtn.addEventListener("click", () => {
    downloadExperimentArtifact("stats_summary.md");
  });
}

initializeTheme();
initializeRandomizeEverySecondSetting();
initializeRotationIncrementSetting();
initializeRandomFlipSetting();
initializeRotate3DSetting();
initializeExtraLineSetting();
initializeArtifactSafeModeSetting();
initializeThemeSelect();
initializeGroundingPresetStudio();
restoreAcquiredData();
restoreSweepPreviewsFromStorage();
if (!restoreClockFaceState()) {
  resetClockFace(true);
} else {
  drawClock();
}
setInterval(() => {
  let changed = false;
  if (randomizeThemeEverySecond) {
    currentClockTheme = getRandomClockTheme();
    changed = true;
  }
  if (rotateClockEverySecond) {
    clockRotationDegrees = (clockRotationDegrees + ROTATION_INCREMENT_DEGREES) % 360;
    changed = true;
  }
  if (randomFlipEverySecond) {
    isFlipped = Math.random() >= 0.5;
    changed = true;
  }
  if (rotateClock3DEverySecond) {
    clock3DRotationDegrees = (clock3DRotationDegrees + ROTATE_3D_INCREMENT_DEGREES) % 360;
    changed = true;
  }
  if (changed) {
    persistClockFaceState();
  }
  drawClock();
}, 1000);
resizeSymbolicOutput();
resizeProbabilisticReasoningOutput();
resizeSymbolicLogOutput();
drawProcessingHistogram();
updateRealtimeLimitInfo();
refreshExperimentsSummary();
setSweepProgress(0, "Idle", "running");
updateSweepStageChecklist("queued", "running");

// Resume sweep progress polling if a sweep was running before page load
void (async () => {
  try {
    const res = await fetch("/sweep-progress");
    const state = await res.json();
    if (state && state.running) {
      setExperimentsBusyState(true);
      const pct = Number.isFinite(Number(state.percent)) ? Number(state.percent) : 0;
      setSweepProgress(pct, `${state.stage} ${Math.floor(pct)}%`, "running");
      updateSweepStageChecklist(state.stage || "running", "running");
      startSweepProgressPolling();
    } else if (state && state.completed) {
      setSweepProgress(100, "Completed", "complete");
      updateSweepStageChecklist("done", "complete");
    } else if (state && state.failed) {
      const pct = Math.max(5, Math.floor(Number(state.percent) || 0));
      setSweepProgress(pct, "Failed", "error");
      updateSweepStageChecklist(state.stage || "error", "error");
    }
  } catch (_) {
    // ignore on load
  }
})();
