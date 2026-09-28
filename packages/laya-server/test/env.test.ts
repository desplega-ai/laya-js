import { ARTIFACT_REPO, ARTIFACT_REVISION } from "@desplega/laya";
import { describe, expect, it } from "vitest";
import { describeEnv, EnvError, loadEnv } from "../src/env.js";

const HOME = { XDG_CACHE_HOME: "/xdg" };

describe("loadEnv", () => {
  it("applies serve.py's defaults plus the plan's additions", () => {
    expect(loadEnv(HOME)).toEqual({
      host: "0.0.0.0",
      port: 8000,
      device: null,
      preload: true,
      models: ["multilingual"],
      defaultModel: "multilingual",
      threads: null,
      autoTask: false,
      maxLoaded: null,
      apiKey: null,
      logLevel: "info",
      maxConcurrent: 16,
      maxTokenBudget: 8192,
      precision: "fp32",
      modelDir: "/models",
      cacheDir: "/xdg/laya-server",
      onnxRepo: ARTIFACT_REPO,
      onnxRevision: ARTIFACT_REVISION,
      hfToken: null,
      hfEndpoint: "https://huggingface.co",
    });
  });

  it("parses every documented variable", () => {
    const env = loadEnv({
      LAYA_HOST: "127.0.0.1",
      LAYA_PORT: " 9000\n",
      LAYA_DEVICE: "cpu",
      LAYA_PRELOAD: "0",
      LAYA_MODELS: "ml, english ,typed",
      LAYA_THREADS: "4",
      LAYA_AUTO_TASK: "yes",
      LAYA_MAX_LOADED: "3",
      LAYA_API_KEY: "k3y",
      LAYA_LOG_LEVEL: "DEBUG",
      LAYA_MAX_CONCURRENT: "4",
      LAYA_MAX_TOKEN_BUDGET: "2048",
      LAYA_PRECISION: "fp32",
      LAYA_MODEL_DIR: "/opt/models",
      LAYA_CACHE_DIR: "/cache",
      LAYA_ONNX_REPO: "someone/mirror",
      LAYA_ONNX_REVISION: "abc123",
      HF_TOKEN: "hf_x",
      HF_ENDPOINT: "http://127.0.0.1:9999/",
    });
    expect(env).toEqual({
      host: "127.0.0.1",
      port: 9000,
      device: "cpu",
      preload: false,
      models: ["multilingual", "english", "typed-decisions"],
      defaultModel: "multilingual",
      threads: 4,
      autoTask: true,
      maxLoaded: 3,
      apiKey: "k3y",
      logLevel: "debug",
      maxConcurrent: 4,
      maxTokenBudget: 2048,
      precision: "fp32",
      modelDir: "/opt/models",
      cacheDir: "/cache",
      onnxRepo: "someone/mirror",
      onnxRevision: "abc123",
      hfToken: "hf_x",
      hfEndpoint: "http://127.0.0.1:9999",
    });
  });

  it("treats empty values as unset", () => {
    const env = loadEnv({ ...HOME, LAYA_MODELS: " , ", LAYA_PORT: "", LAYA_API_KEY: "", LAYA_THREADS: " " });
    expect(env.models).toEqual(["multilingual"]);
    expect(env.port).toBe(8000);
    expect(env.apiKey).toBeNull();
    expect(env.threads).toBeNull();
  });

  it("defaults to the first of LAYA_MODELS", () => {
    expect(loadEnv({ LAYA_MODELS: "english,multilingual" }).defaultModel).toBe("english");
  });

  it("dedupes aliased checkpoints", () => {
    expect(loadEnv({ LAYA_MODELS: "multilingual,ml,multi" }).models).toEqual(["multilingual"]);
  });

  it("keeps the API key byte for byte", () => {
    expect(loadEnv({ LAYA_API_KEY: " spaced " }).apiKey).toBe(" spaced ");
  });

  const bad: Array<[string, string, RegExp]> = [
    ["LAYA_PORT", "abc", /LAYA_PORT.*1-65535/],
    ["LAYA_PORT", "0", /LAYA_PORT/],
    ["LAYA_PORT", "65536", /LAYA_PORT/],
    ["LAYA_PORT", "80.5", /LAYA_PORT/],
    ["LAYA_PRECISION", "int8", /LAYA_PRECISION.*fp32/],
    ["LAYA_MODELS", "multilingual,gpt-4", /LAYA_MODELS entry "gpt-4"/],
    ["LAYA_THREADS", "0", /LAYA_THREADS/],
    ["LAYA_THREADS", "two", /LAYA_THREADS/],
    ["LAYA_MAX_LOADED", "-1", /LAYA_MAX_LOADED/],
    ["LAYA_MAX_CONCURRENT", "0", /LAYA_MAX_CONCURRENT/],
    ["LAYA_MAX_CONCURRENT", "16x", /LAYA_MAX_CONCURRENT/],
    ["LAYA_MAX_TOKEN_BUDGET", "1e4", /LAYA_MAX_TOKEN_BUDGET/],
    ["LAYA_PRELOAD", "maybe", /LAYA_PRELOAD/],
    ["LAYA_AUTO_TASK", "2", /LAYA_AUTO_TASK/],
    ["LAYA_LOG_LEVEL", "loud", /LAYA_LOG_LEVEL/],
    ["HF_ENDPOINT", "ftp://example.com", /HF_ENDPOINT/],
  ];
  it.each(bad)("fails fast on %s=%j", (name, value, message) => {
    expect(() => loadEnv({ [name]: value })).toThrow(EnvError);
    expect(() => loadEnv({ [name]: value })).toThrow(message);
  });
});

describe("describeEnv", () => {
  it("never includes secret values", () => {
    const text = JSON.stringify(describeEnv(loadEnv({ LAYA_API_KEY: "api-secret-1", HF_TOKEN: "hf_secret_2" })));
    expect(text).not.toContain("api-secret-1");
    expect(text).not.toContain("hf_secret_2");
    expect(text).toContain('"apiKey":"(set)"');
    expect(text).toContain('"hfToken":"(set)"');
  });
});
