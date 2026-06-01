// Headless smoke test for the demo UI's run flow (would have caught the
// applyLang-wiped-#speed bug that broke every Run in the browser).
//
//   pip install flask duckdb && python demo/app.py     # in one terminal
//   npm install jsdom && node demo/test_ui.js          # in another
//
// Loads the served page, runs its inline script in jsdom, clicks Run, and
// asserts the pipeline rendered with zero JS errors.
const { JSDOM } = require('jsdom');

(async () => {
  const base = 'http://127.0.0.1:5000/';
  const html = await (await fetch(base)).text();
  const errors = [];
  const dom = new JSDOM(html, {
    url: base, runScripts: 'dangerously', pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = (u, o) => fetch(new URL(u, base), o);
      w.alert = (m) => errors.push('ALERT: ' + m);
      w.localStorage = { getItem: () => null, setItem: () => {} };
      w.addEventListener('error', (e) => errors.push('ERR: ' + (e.error && e.error.stack || e.message)));
      w.onerror = (msg, s, l, c, err) => errors.push('ONERR: ' + (err && err.stack || msg));
    },
  });
  const w = dom.window, $ = (id) => w.document.getElementById(id);
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  await sleep(1500);                       // init: fetch content/usecases + reset

  const checks = [
    ['#speed present (not wiped by applyLang)', !!$('speed')],
    ['mode chips built', $('modechips').children.length === 6],
    ['use cases rendered', $('usecases').children.length === 5],
  ];
  if ($('speed')) $('speed').value = '60';
  $('run').click();
  await sleep(2500);
  checks.push(['Run rendered the pipeline', $('steps').children.length > 0]);
  checks.push(['no JS errors during run', errors.length === 0]);

  let ok = true;
  for (const [name, pass] of checks) { console.log((pass ? 'PASS' : 'FAIL') + '  ' + name); ok = ok && pass; }
  errors.slice(0, 5).forEach(e => console.log('  ' + e.slice(0, 160)));
  console.log(ok ? '\nALL OK' : '\nFAILED');
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error('harness error:', e); process.exit(1); });
