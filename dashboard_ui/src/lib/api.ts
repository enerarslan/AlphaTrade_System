import axios from "axios";

const defaultApiBaseUrl = (() => {
  if (typeof window === "undefined") {
    return "http://127.0.0.1:8000";
  }
  const isLocalDev =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.port === "5173";
  return isLocalDev ? "http://127.0.0.1:8000" : `${window.location.origin}/api`;
})();

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? defaultApiBaseUrl;

let accessToken: string | null = localStorage.getItem("alphatrade_token");

export function setAccessToken(token: string) {
  accessToken = token;
  localStorage.setItem("alphatrade_token", token);
}

export function clearAccessToken() {
  accessToken = null;
  localStorage.removeItem("alphatrade_token");
}

export function getAccessToken() {
  return accessToken;
}

export function buildApiUrl(path: string, query?: Record<string, string | number | boolean | undefined>) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = String(apiBaseUrl).trim();
  const url = /^https?:\/\//i.test(base)
    ? new URL(normalizedPath, `${base.replace(/\/+$/, "")}/`)
    : new URL(`${base.replace(/\/+$/, "")}${normalizedPath}`, window.location.origin);

  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined || value === null) {
        continue;
      }
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

export const api = axios.create({
  baseURL: apiBaseUrl,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.request.use((config) => {
  if (accessToken) {
    config.headers.Authorization = `Bearer ${accessToken}`;
  }
  return config;
});
