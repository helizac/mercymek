/* Detection loss must never remove a physical actor. */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {actorVisible} from '../frontend/js/replay-pose.js';

const v={model:{},persistent:true,observed_range:[1,4]};
assert.equal(actorVisible(v,.99),true);
assert.equal(actorVisible(v,1),true);
assert.equal(actorVisible(v,2.5),true); // An internal detection gap retains the identity.
assert.equal(actorVisible(v,4),true);
assert.equal(actorVisible(v,4.01),true);
assert.equal(actorVisible({...v,existence_range:[0,10]},5),true);
assert.equal(actorVisible({model:{},persistent:true,key:'ego'},10),true);
assert.equal(actorVisible({visible_range:[1,4]},5),false); // Legacy photo layers.

const {moments}=JSON.parse(readFileSync(new URL('../data/cases/hero/moments.json',import.meta.url)));
for(const m of moments){
 const at=(key,time)=>actorVisible(m.vehicles.find(v=>v.key===key),time-m.t_frame);
 assert.equal(at('car:3',5.5),true,'the real early observation must remain');
 assert.equal(at('car:3',8),true,'a departed vehicle must remain in the world');
 assert.equal(at('car:3',10.95),true,'exit correction must change motion, not visibility');
 assert.equal(at('car:16',10.95),true,'the real distant white car must remain');
 assert.equal(at('car:7',8.13),true,'the incident car must remain at the reviewed impact frame');
 assert.equal(at('car:10',10.95),true,'another detection loss must retain its physical actor');
}
console.log('Persistent actor visibility through detection boundaries, source exits and impact PASS');
