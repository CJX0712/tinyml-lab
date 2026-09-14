const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx = { console, Math, Float64Array, Object, Array, JSON, isFinite, Infinity };
ctx.globalThis = ctx; vm.createContext(ctx); vm.runInContext(m[1], ctx);
const TML = ctx.TML;

// Train on spiral, dump decision grid as ASCII + loss trajectory
const net = new TML.Net([2, 16, 16, 1], 20260914);
const ds = TML.makeDS('spiral', 220, 42);
const traj = [];
for (let i = 0; i < 800; i++) traj.push(net.trainEpoch(ds, 0.3));

const R = 40, lines = [];
for (let j = 0; j < R; j++) {
  let row = '';
  for (let i = 0; i < R; i++) {
    const x = (i / (R - 1)) * 2 - 1, y = 1 - (j / (R - 1)) * 2;
    const p = net.predict([x, y]);
    row += p > 0.85 ? '#' : p > 0.6 ? '+' : p > 0.4 ? '.' : p > 0.15 ? '-' : ' ';
  }
  lines.push(row);
}
// overlay points
for (let k = 0; k < ds.X.length; k++) {
  const i = Math.round((ds.X[k][0] + 1) / 2 * (R - 1)),
        j = Math.round((1 - (ds.X[k][1] + 1) / 2) * (R - 1));
  const c = lines[j].split('');
  c[i] = ds.y[k] ? 'O' : 'o';
  lines[j] = c.join('');
}

const out = [];
out.push('spiral decision map (40x40)  #=p>.85 +=.6-0.85 .=.4-.6 -=.15-.4 space=<.15  O=class1 o=class0');
out.push(lines.join('\n'));
out.push('');
out.push('loss trajectory (every 100th epoch):');
for (let i = 0; i < traj.length; i += 100) out.push(`  step ${String(i).padStart(4)}  loss ${traj[i].toFixed(5)}`);
out.push(`  final acc = ${(net.accuracy(ds) * 100).toFixed(1)}%`);

// also dump xor map (should be clean checkerboard)
const nx = new TML.Net([2, 8, 8, 1], 20260914);
const dx = TML.makeDS('xor', 220, 42);
for (let i = 0; i < 400; i++) nx.trainEpoch(dx, 0.3);
const xl = [];
for (let j = 0; j < 24; j++) {
  let row = '';
  for (let i = 0; i < 24; i++) {
    const x = (i / 23) * 2 - 1, y = 1 - (j / 23) * 2;
    const p = nx.predict([x, y]);
    row += p > 0.5 ? '#' : '.';
  }
  xl.push(row);
}
out.push('\nxor decision map (24x24, #=class1):');
out.push(xl.join('\n'));

fs.writeFileSync(path.join(__dirname, '_probe.txt'), out.join('\n') + '\n');
console.log('probe written');
