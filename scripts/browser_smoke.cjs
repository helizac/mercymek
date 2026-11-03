/* Real browser smoke test for the local reconstruction viewer. */
const { chromium } = require('../data/browser-tests/node_modules/playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const browser = await chromium.launch({ channel:'msedge', headless:true, args:['--enable-webgl', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport:{width:1600,height:1000} });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
  await page.goto('http://localhost:8000/?case=cutin_rearend&view=scene&t=3.5');
  await page.waitForTimeout(15000);
  await page.screenshot({path:'data/research/viewer-cutin.png'});
  console.log(JSON.stringify({title:await page.title(), hud:await page.locator('#hud3').innerText(), errors},null,2));
  await browser.close();
  if (errors.length) process.exitCode = 1;
})();
