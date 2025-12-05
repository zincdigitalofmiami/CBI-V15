#!/usr/bin/env node

const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const SESSION_ID = "debug-session";
const RUN_ID = "dataform-core-run1";
const LOG_PATH = "/Volumes/Satechi Hub/CBI-V15/.cursor/debug.log";

function logDebug({ hypothesisId, location, message, data }) {
  const payload = {
    sessionId: SESSION_ID,
    runId: RUN_ID,
    hypothesisId,
    location,
    message,
    data,
    timestamp: Date.now()
  };

  // #region agent log
  try {
    // Prefer HTTP ingestion when fetch is available (Node 18+)
    if (typeof fetch === "function") {
      fetch("http://127.0.0.1:7242/ingest/7e84ff69-2ec3-4546-aa0d-a5191b254e43", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).catch(() => {});
    }
  } catch (_) {
    // ignore fetch issues, we always fall back to file logging below
  }

  try {
    fs.appendFileSync(LOG_PATH, JSON.stringify(payload) + "\n", { encoding: "utf8" });
  } catch (_) {
    // best-effort only
  }
  // #endregion
}

async function main() {
  logDebug({
    hypothesisId: "H_env",
    location: "scripts/diagnose_dataform_core.js:25",
    message: "Environment before Dataform run",
    data: {
      cwd: process.cwd(),
      nodeVersion: process.version,
      envPath: process.env.PATH || ""
    }
  });

  const resolution = {};

  try {
    const resolved = require.resolve("@dataform/core");
    resolution.root = { ok: true, resolved };
  } catch (err) {
    resolution.root = { ok: false, error: err.message };
  }

  try {
    const resolvedLocal = require.resolve("@dataform/core", {
      paths: [path.join(__dirname, "..", "dataform", "node_modules")]
    });
    resolution.dataform = { ok: true, resolved: resolvedLocal };
  } catch (err) {
    resolution.dataform = { ok: false, error: err.message };
  }

  logDebug({
    hypothesisId: "H1",
    location: "scripts/diagnose_dataform_core.js:48",
    message: "@dataform/core resolution check",
    data: resolution
  });

  const dataformCwd = path.join(__dirname, "..", "dataform");
  const cmd = process.platform === "win32" ? "npx.cmd" : "npx";
  const args = ["dataform", "compile"];

  logDebug({
    hypothesisId: "H2",
    location: "scripts/diagnose_dataform_core.js:59",
    message: "Spawning Dataform CLI",
    data: { cmd, args, cwd: dataformCwd }
  });

  const child = spawn(cmd, args, {
    cwd: dataformCwd,
    stdio: "inherit"
  });

  child.on("exit", (code, signal) => {
    logDebug({
      hypothesisId: "H3",
      location: "scripts/diagnose_dataform_core.js:70",
      message: "Dataform CLI exit",
      data: { code, signal }
    });
    process.exit(code ?? 1);
  });
}

main().catch((err) => {
  logDebug({
    hypothesisId: "H_unhandled",
    location: "scripts/diagnose_dataform_core.js:80",
    message: "Unhandled error in diagnose_dataform_core",
    data: { message: err.message, stack: err.stack }
  });
  process.exit(1);
});


