/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_WS_URL: string;
  readonly VITE_NODE_ENV: string;
  readonly VITE_ENABLE_DEBUG: boolean;
  readonly VITE_ENABLE_AGENT_PIPELINE: boolean;
  readonly VITE_PIPELINE_POLL_INTERVAL: string;
  readonly VITE_SHOW_AGENT_DETAILS: boolean;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
