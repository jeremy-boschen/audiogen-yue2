import {defineConfig, devices} from '@playwright/test';

// The explorer pages, built from this machine's microscope runs and served the
// way people open them (bin/explore.py serve), then driven in Chromium.
// EXPLORER_URL points the same tests at a published copy instead, in Chromium
// and WebKit: the check after publishing (docs/microscope-explorer.md).
const PORT = 8779;
const live = process.env.EXPLORER_URL;

export default defineConfig({
  testDir: 'tests/pages',
  timeout: 60_000,
  workers: 1,               // one GPU-backed page at a time: frame counts stay honest
  reporter: [['list']],
  use: {baseURL: live ?? `http://127.0.0.1:${PORT}/00_explore/`,
        launchOptions: {args: ['--autoplay-policy=no-user-gesture-required']}},
  projects: live
    ? [{name: 'chromium', use: {...devices['Desktop Chrome']}}, {name: 'webkit', use: {...devices['Desktop Safari'], launchOptions: {}}}]
    : [{name: 'chromium', use: {browserName: 'chromium'}}],
  webServer: live ? undefined : {
    command: `.venv-dev/bin/python bin/explore.py all && .venv-dev/bin/python bin/explore.py serve --no-open --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/00_explore/index.html`,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: 'ignore',
  },
});
