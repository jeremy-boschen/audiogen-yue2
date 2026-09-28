import {expect, test, type Page} from '@playwright/test';

/*
 * The explorer pages draw a bloom-lit WebGL scene. They must draw only while
 * something moves: a page left open, even unfocused, once held a core and the
 * GPU for hours (common.js, EX.stage). Frames are counted as WebGL draw calls,
 * wrapped before any page script runs.
 */

declare global {
  interface Window { __draws: number; __focused?: boolean }
}

async function countDraws(page: Page, focused?: boolean) {
  await page.addInitScript((focus) => {
    window.__draws = 0;
    if (focus !== undefined) {
      window.__focused = focus;
      document.hasFocus = () => window.__focused!;
    }
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, ...args: Parameters<typeof original>) {
      const gl = original.apply(this, args) as WebGL2RenderingContext | null;
      if (gl && typeof args[0] === 'string' && args[0].startsWith('webgl') && !(gl as unknown as {__counted?: true}).__counted) {
        (gl as unknown as {__counted?: true}).__counted = true;
        for (const name of ['drawArrays', 'drawElements', 'drawArraysInstanced', 'drawElementsInstanced'] as const) {
          const draw = (gl as unknown as Record<string, (...a: unknown[]) => unknown>)[name];
          if (draw) (gl as unknown as Record<string, unknown>)[name] = (...a: unknown[]) => { window.__draws++; return draw.apply(gl, a); };
        }
      }
      return gl;
    } as typeof original;
  }, focused);
}

/** The map is built only for this machine; a published copy has focus and stack. */
const published = !!process.env.EXPLORER_URL;
const PAGES = published ? ['focus.html', 'stack.html'] : ['focus.html', 'stack.html', 'map.html'];

const draws = (page: Page) => page.evaluate(() => window.__draws);

/** Draw calls made over `ms`. */
async function drawnOver(page: Page, ms: number) {
  const before = await draws(page);
  await page.waitForTimeout(ms);
  return (await draws(page)) - before;
}

/** Waits until a whole second passes without a frame, and says how long that took. */
async function settles(page: Page, within = 10_000) {
  const start = Date.now();
  while (Date.now() - start < within) {
    if ((await drawnOver(page, 1000)) === 0) return Date.now() - start;
  }
  throw new Error(`still drawing after ${within} ms`);
}

/** Opens a page and fails on any script error or console error. */
async function open(page: Page, name: string) {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  // Cloudflare's proxy injects its analytics beacon into the published pages; it is not ours.
  page.on('console', m => { if (m.type() === 'error' && !m.location().url.includes('cloudflareinsights.com')) errors.push(m.text()); });
  await page.goto(name);
  await expect(page.locator('#stage canvas')).toBeVisible();
  return errors;
}

for (const name of PAGES) {
  test(`${name} opens cleanly, draws, then stops drawing when left alone`, async ({page}) => {
    await countDraws(page, false);
    const errors = await open(page, name);
    await expect.poll(() => draws(page)).toBeGreaterThan(0);
    await settles(page);
    expect(await drawnOver(page, 2000)).toBe(0);
    expect(errors).toEqual([]);
  });

  test(`${name} wakes on input, then goes quiet again`, async ({page}) => {
    await countDraws(page, false);
    await open(page, name);
    await settles(page);
    await page.mouse.move(400, 300);
    await page.mouse.move(420, 320);
    expect(await drawnOver(page, 500)).toBeGreaterThan(0);
    await settles(page, 6000);
  });
}

test('the map turns on its own only while its window has focus', async ({page}) => {
  test.skip(published, 'the map is not published');
  await countDraws(page, true);
  await open(page, 'map.html');
  await page.waitForTimeout(5000);
  expect(await drawnOver(page, 1000)).toBeGreaterThan(0);          // focused: the ambient turn goes on
  await page.evaluate(() => { window.__focused = false; });
  await settles(page, 6000);                                         // unfocused: it stops
  await page.evaluate(() => { window.__focused = true; window.dispatchEvent(new Event('focus')); });
  expect(await drawnOver(page, 1000)).toBeGreaterThan(0);          // back: it turns again
});

test('playing keeps the stack drawing, and the end of the phrase lets it stop', async ({page}) => {
  await countDraws(page, false);
  await open(page, 'stack.html');
  await settles(page);
  await page.keyboard.press(' ');                                    // the first phrase: about 5.2 s of sound
  await page.waitForTimeout(3200);                                   // past the 2.5 s a key press alone keeps it awake
  expect(await drawnOver(page, 1000)).toBeGreaterThan(0);
  await settles(page, 12_000);
});

test('playing keeps the focus view drawing, even when the sound starts after the click', async ({page}) => {
  await countDraws(page, false);
  await open(page, 'focus.html');
  await settles(page);
  await page.locator('#play').click();
  await expect(page.locator('#play')).not.toHaveText(/▶/, {timeout: 20_000});
  await page.waitForTimeout(3500);
  expect(await drawnOver(page, 1000)).toBeGreaterThan(0);
  await page.locator('#play').click();
  await settles(page);
});
