/* README screenshots of the running demo (server on :8000). Playwright lives in data/browser-tests (not in the repo).
   node scripts/capture_screens.cjs [case]   ->  docs/img/{video,scene,scene-orbit,report,map,evidence}.png */
const { chromium } = require('../data/browser-tests/node_modules/playwright');
const caseId = process.argv[2] || 'hero';
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--enable-webgl', '--ignore-gpu-blocklist', '--use-angle=d3d11'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const shot = async (name, wait = 4000) => { await page.waitForTimeout(wait); await page.screenshot({ path: `docs/img/${name}.png` }); console.log('saved', name); };
  await page.goto(`http://localhost:8000/?case=${caseId}&view=video&t=6`);
  await shot('video', 6000);
  await page.goto(`http://localhost:8000/?case=${caseId}&view=scene&t=8`);
  await shot('scene', 22000);
  const box = await page.locator('#v-scene').boundingBox();
  if (box) {                       // orbit the free camera a little to show it is a real 3D world
    await page.mouse.move(box.x + box.width * .55, box.y + box.height * .5);
    await page.mouse.down(); await page.mouse.move(box.x + box.width * .38, box.y + box.height * .46, { steps: 20 }); await page.mouse.up();
    await shot('scene-orbit', 2500);
  }
  for (const v of ['report', 'map', 'evidence']) {
    await page.goto(`http://localhost:8000/?case=${caseId}&view=${v}&t=8`);
    await shot(v, v === 'map' ? 8000 : 5000);
  }
  await browser.close();
})();
