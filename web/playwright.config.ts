import { defineConfig, devices } from '@playwright/test'

// Separate ports from `make dev` so the e2e run never touches a live session.
const API_PORT = 8417
const WEB_PORT = 5417
const uv = process.env.UV ?? 'uv'

export default defineConfig({
  testDir: './e2e',
  timeout: 180_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    ...devices['Desktop Chrome'],
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      // Study mode reads a pack built from the committed Pluribus fixture.
      command:
        `${uv} run python -m strategy.data.study --source fixture --out artifacts/e2e-study/pluribus-sample && ` +
        `${uv} run uvicorn --factory arena.api.app:create_app --host 127.0.0.1 --port ${API_PORT}`,
      cwd: '..',
      url: `http://127.0.0.1:${API_PORT}/health`,
      // Deterministic clock: human deadlines move only through POST /__test__/clock/advance.
      env: {
        ARENA_FAKE_CLOCK: '1',
        ARENA_BOT_DELAY_MS: '0',
        ARENA_HAND_PAUSE_MS: '0',
        ARENA_OPERATOR_TOKEN: 'e2e-operator-token',
        ARENA_ALLOWED_ORIGINS: `http://127.0.0.1:${WEB_PORT}`,
        ARENA_STUDY_DIR: 'artifacts/e2e-study',
        ARENA_ADVISOR_DIR: 'artifacts/e2e-advisor',
      },
      reuseExistingServer: false,
    },
    {
      command: 'npx vite',
      url: `http://127.0.0.1:${WEB_PORT}`,
      env: { ARENA_API_PORT: String(API_PORT), ARENA_WEB_PORT: String(WEB_PORT) },
      reuseExistingServer: false,
    },
  ],
})
