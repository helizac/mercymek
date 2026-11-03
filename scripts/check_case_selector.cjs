/* Responsive chooser, full case coverage, keyboard exit and real case loading. */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const base=process.env.MERCY_TEST_URL||'http://localhost:8001';
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const directory=process.env.MERCY_TEST_ARTIFACTS||'data/research/ui-vehicles';fs.mkdirSync(directory,{recursive:true});
 try{
  const page=await browser.newPage(),errors=[],reports=[];page.on('pageerror',e=>errors.push(e.message));
  const response=await page.request.get(`${base}/api/cases`);assert(response.ok());
  const inventory=await response.json();
  const expected=(process.env.MERCY_EXPECT_CASES||'hero,cutin_rearend,lanechange,night_rain,snow_pileup').split(',');
  for(const id of expected)assert(inventory.some(c=>c.id===id),`Missing expected case ${id}`);
  for(const [width,height] of [[1920,1080],[1113,768],[960,600],[390,844]]){
   await page.setViewportSize({width,height});
   await page.goto(`${base}/?case=cutin_rearend&view=video`);
   await page.waitForFunction(()=>window.MERCY?.feed.case?.id==='cutin_rearend');
   await page.waitForFunction(()=>Number(getComputedStyle(document.querySelector('#boot')).opacity)<.01);
   await page.locator('#caseBtn').click();
   await page.waitForFunction(()=>Number(getComputedStyle(document.querySelector('#casesel')).opacity)>.99);
   await page.evaluate(()=>Promise.all(document.querySelector('#casesel').getAnimations().map(a=>a.finished)));
   await page.evaluate(()=>Promise.all([...document.querySelectorAll('.case-poster')].map(p=>{
    const url=getComputedStyle(p).backgroundImage.slice(5,-2);if(!url)return;
    const image=new Image();image.src=url;return image.decode();
   })));
   const result=await page.evaluate(()=>{
    const box=document.querySelector('#casesel'),cards=[...box.querySelectorAll('[data-case-id]')];
    const rects=cards.map(c=>c.getBoundingClientRect().toJSON());
    let overlaps=0;
    for(let a=0;a<rects.length;a++)for(let b=0;b<a;b++){
     const A=rects[a],B=rects[b];if(Math.min(A.right,B.right)>Math.max(A.left,B.left)&&Math.min(A.bottom,B.bottom)>Math.max(A.top,B.top))overlaps++;
    }
    return {ids:cards.map(c=>c.dataset.caseId),overlaps,current:box.querySelector('[aria-current]')?.dataset.caseId,
     collisions:[...box.querySelectorAll('.case-event')].filter(c=>c.textContent==='Collision').length,
     nearMisses:[...box.querySelectorAll('.case-event')].filter(c=>c.textContent==='Near miss').length,
     horizontalOverflow:box.querySelector('.case-dialog').scrollWidth>box.querySelector('.case-dialog').clientWidth,
     labelsFit:cards.every(c=>{const r=c.getBoundingClientRect(),t=c.querySelector('.cap').getBoundingClientRect();return t.left>=r.left&&t.right<=r.right&&t.bottom<=r.bottom;})};
   });
   assert.deepEqual(new Set(result.ids),new Set(inventory.map(c=>c.id)));
   assert.equal(result.overlaps,0);assert.equal(result.current,'cutin_rearend');assert(!result.horizontalOverflow);assert(result.labelsFit);
   assert.equal(result.collisions,inventory.filter(c=>c.event_kind==='collision').length);
   assert.equal(result.nearMisses,inventory.filter(c=>c.event_kind==='near-miss').length);
   await page.screenshot({path:`${directory}/case-selector-${width}.png`});
   await page.locator('.case-close').focus();await page.keyboard.press('Shift+Tab');
   assert.equal(await page.evaluate(()=>document.activeElement.dataset.caseId),result.ids.at(-1));
   await page.keyboard.press('Tab');assert(await page.locator('.case-close').evaluate(e=>e===document.activeElement));
   await page.keyboard.press('Escape');assert(await page.locator('#casesel').isHidden());
   assert.equal(await page.evaluate(()=>document.activeElement.id),'caseBtn');
   await page.locator('#caseBtn').click();await page.locator('.case-close').click();assert(await page.locator('#casesel').isHidden());
   await page.locator('#caseBtn').click();await page.locator('#casesel').click({position:{x:2,y:2}});
   assert(await page.locator('#casesel').isHidden());
   reports.push({width,height,...result});
  }
  await page.setViewportSize({width:1600,height:1000});
  await page.locator('#caseBtn').click();await page.locator('[data-case-id="hero"]').click();
  await page.waitForFunction(()=>MERCY.feed.case.id==='hero');assert(await page.locator('#casesel').isHidden());
  assert.deepEqual(errors,[]);
  fs.writeFileSync(`${directory}/case-selector.json`,JSON.stringify({reports,selected_case:'hero',console_errors:errors},null,2));
  console.log(`All ${inventory.length} cases visible; ${reports.length} viewport layouts; event labels; close/Escape/focus and case switching PASS`);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
