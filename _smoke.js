const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) { console.error('engine script not found'); process.exit(1); }
const ctx = { console, Math, Float64Array, Object, Array, JSON, isFinite, Infinity, NaN };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const TML = ctx.TML;

let pass = 0, fail = 0; const fails = [];
function assert(name, cond) {
  if (cond) { pass++; }
  else { fail++; fails.push(name); }
}

// 1) determinism: same seed -> identical predictions
{
  const a = new TML.Net([2, 8, 8, 1], 7), b = new TML.Net([2, 8, 8, 1], 7);
  let same = true;
  for (let i = 0; i < 40; i++) {
    const x = [Math.sin(i) * 0.9, Math.cos(i * 1.7) * 0.9];
    if (a.predict(x) !== b.predict(x)) { same = false; break; }
  }
  assert('determinism same seed', same);
  const c = new TML.Net([2, 8, 8, 1], 8);
  assert('different seed differs', c.predict([0.3, -0.2]) !== a.predict([0.3, -0.2]));
}

// 2) dataset invariants
{
  const kinds = ['xor', 'circle', 'moons', 'spiral'];
  for (const k of kinds) {
    const ds = TML.makeDS(k, 200, 42);
    assert('ds ' + k + ' length', ds.X.length === 200 && ds.y.length === 200);
    let ok = true, ones = 0;
    for (let i = 0; i < 200; i++) {
      if (typeof ds.y[i] !== 'number' || (ds.y[i] !== 0 && ds.y[i] !== 1)) ok = false;
      if (Math.abs(ds.X[i][0]) > 1.4 || Math.abs(ds.X[i][1]) > 1.4) ok = false;
      if (ds.y[i] === 1) ones++;
    }
    assert('ds ' + k + ' labels+range', ok);
    assert('ds ' + k + ' has both classes', ones > 10 && ones < 190);
  }
  const d1 = TML.makeDS('xor', 50, 42), d2 = TML.makeDS('xor', 50, 42);
  assert('ds deterministic', JSON.stringify(d1) === JSON.stringify(d2));
}

// 3) gradient check (analytic == central difference)
{
  const g = TML.gradCheck([2, 4, 3, 1], 5, 1e-5);
  assert('gradcheck maxRelErr < 1e-6 (got ' + g.maxRelErr.toExponential(2) + ' @' + g.worst + ')', g.maxRelErr < 1e-6);
}

// 4) XOR convergence: loss decreases, 4/4 centers classified
{
  const net = new TML.Net([2, 8, 8, 1], 20260914);
  const ds = TML.makeDS('xor', 220, 42);
  const l0 = net.trainEpoch(ds, 0.3);
  let loss = l0;
  for (let i = 0; i < 400; i++) loss = net.trainEpoch(ds, 0.3);
  assert('xor loss decreases (' + l0.toFixed(3) + ' -> ' + loss.toFixed(4) + ')', loss < l0 && loss < 0.05);
  const quad = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  let allOK = true;
  for (const [x, y] of quad) {
    const p = net.predict([x * 0.7, y * 0.7]);
    if ((p > 0.5 ? 1 : 0) !== (x * y > 0 ? 1 : 0)) allOK = false;
  }
  assert('xor 4/4 quadrant classified', allOK);
  assert('xor accuracy >= 0.98 (got ' + net.accuracy(ds) + ')', net.accuracy(ds) >= 0.98);
}

// 5) other datasets reach high accuracy
{
  const cfg = [['circle', 300], ['moons', 400], ['spiral', 800]];
  for (const [kind, steps] of cfg) {
    const net = new TML.Net([2, 16, 16, 1], 20260914);
    const ds = TML.makeDS(kind, 220, 42);
    let loss = 1;
    for (let i = 0; i < steps; i++) loss = net.trainEpoch(ds, 0.3);
    const acc = net.accuracy(ds);
    assert(kind + ' acc >= 0.97 (got ' + acc.toFixed(3) + ', loss ' + loss.toFixed(4) + ')', acc >= 0.97);
  }
}

// 6) predict range & extreme stability
{
  const net = new TML.Net([2, 8, 8, 1], 3);
  let inRange = true;
  for (let i = 0; i < 200; i++) {
    const p = net.predict([Math.sin(i * 12.9) * 5, Math.cos(i * 3.1) * 5]); // far outside training range
    if (!(p >= 0 && p <= 1) || !isFinite(p)) { inRange = false; break; }
  }
  assert('sigmoid output in [0,1] even far OOD', inRange);
}

// 7) grid shape & values
{
  const net = new TML.Net([2, 8, 8, 1], 11);
  const g = TML.grid(net, 16);
  let ok = g.length === 256, inR = true;
  for (let i = 0; i < 256; i++) if (!(g[i] >= 0 && g[i] <= 1)) inR = false;
  assert('grid size 16x16', ok);
  assert('grid values in [0,1]', inR);
}

// 8) empty/tiny edge cases
{
  const ds = TML.makeDS('xor', 2, 42); // 1 per class
  const net = new TML.Net([2, 4, 1], 9);
  const l = net.trainEpoch(ds, 0.1);
  assert('single-sample-per-class trains finitely', isFinite(l));
}

const summary = `PASS ${pass} / ${pass + fail}\n` + (fail ? 'FAIL: ' + fails.join(' | ') : 'ALL GREEN') + '\n';
fs.writeFileSync(path.join(__dirname, '_smoke.log'), summary);
console.log(summary);
process.exit(fail ? 1 : 0);
