/* Headless-Chrome frame renderer driven over CDP with zero npm deps.
   usage:  node render.js <workDir> <outputMp4>
           node render.js <workDir> --probe 1,7,14   -> single PNG frames to <workDir>/probe/

   <workDir> must already contain: index.html, primitives.js, timing.js,
   scenes.js, and the narration audio file named in timing.json's
   narrationFile field - see video_scene.py, which assembles all of this
   per day before invoking this script. */
const {spawn, execFileSync} = require('child_process');
const fs = require('fs');
const path = require('path');

const FPS = 30;
const PORT = 9333 + Math.floor(Math.random() * 500); // avoid clashing with a concurrent render

const workDir = process.argv[2] && path.resolve(process.argv[2]);
if (!workDir) {
  console.error('usage: node render.js <workDir> <outputMp4|--probe t1,t2,...>');
  process.exit(1);
}
const probeArg = process.argv.indexOf('--probe');
const probes = probeArg > -1 ? process.argv[probeArg + 1].split(',').map(Number) : null;
const outputMp4 = probes ? null : path.resolve(process.argv[3]);

const TIMING = JSON.parse(fs.readFileSync(path.join(workDir, 'timing.json'), 'utf8'));
const DURATION = TIMING.duration;

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium-browser',
    '/usr/bin/chromium',
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  throw new Error('No Chrome/Chromium binary found - set CHROME_PATH env var.');
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function launchChrome() {
  const CHROME = findChrome();
  const profile = path.join(workDir, '.chrome-profile');
  fs.rmSync(profile, {recursive: true, force: true});
  const proc = spawn(
    CHROME,
    [
      '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
      '--disable-extensions', '--mute-audio', '--hide-scrollbars', '--no-sandbox',
      '--force-device-scale-factor=1', '--window-size=1280,760',
      `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
      'file://' + path.join(workDir, 'index.html'),
    ],
    {stdio: ['ignore', 'ignore', 'pipe']}
  );
  proc.stderr.on('data', (d) => {
    const s = d.toString();
    if (/ERROR|FATAL/.test(s)) process.stderr.write('[chrome] ' + s);
  });

  for (let i = 0; i < 120; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === 'page' && t.url.includes('index.html'));
      if (page?.webSocketDebuggerUrl) return {proc, wsUrl: page.webSocketDebuggerUrl};
    } catch {}
    await sleep(250);
  }
  throw new Error('Chrome did not expose a page target');
}

function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const pending = new Map();
    let id = 0;
    ws.addEventListener('message', (ev) => {
      const m = JSON.parse(ev.data);
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      m.error ? p.reject(new Error(JSON.stringify(m.error))) : p.resolve(m.result);
    });
    ws.addEventListener('error', (e) => reject(new Error('ws error: ' + (e.message || 'unknown'))));
    ws.addEventListener('open', () =>
      resolve({
        ws,
        send: (method, params = {}) =>
          new Promise((res, rej) => {
            const mid = ++id;
            pending.set(mid, {resolve: res, reject: rej});
            ws.send(JSON.stringify({id: mid, method, params}));
          }),
      })
    );
  });
}

async function evaluate(cdp, expression) {
  const r = await cdp.send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'page exception');
  return r.result.value;
}

async function main() {
  const {proc, wsUrl} = await launchChrome();
  const cdp = await connect(wsUrl);
  let ready = false;
  for (let i = 0; i < 80; i++) {
    if (await evaluate(cdp, 'window.ready === true')) {
      ready = true;
      break;
    }
    await sleep(150);
  }
  if (!ready) {
    const err = await evaluate(cdp, 'window.pageError || "(no error captured, page just never set window.ready)"');
    throw new Error('Page never became ready: ' + err);
  }
  await evaluate(cdp, 'document.fonts.ready.then(()=>1)');

  const grab = async (t) =>
    Buffer.from(await evaluate(cdp, `drawFrame(${t.toFixed(4)}); C.toDataURL('image/png').slice(22)`), 'base64');

  if (probes) {
    const probeDir = path.join(workDir, 'probe');
    fs.mkdirSync(probeDir, {recursive: true});
    for (const t of probes) {
      fs.writeFileSync(path.join(probeDir, `t${t}.png`), await grab(t));
      console.log('probe', t, 's');
    }
    cdp.ws.close();
    proc.kill();
    return;
  }

  const total = Math.round(DURATION * FPS);
  const narrationPath = path.join(workDir, TIMING.narrationFile);
  const ff = spawn(
    'ffmpeg',
    [
      '-y', '-loglevel', 'error',
      '-f', 'image2pipe', '-framerate', String(FPS), '-i', 'pipe:0',
      '-i', narrationPath,
      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'medium',
      '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart',
      '-t', String(DURATION), outputMp4,
    ],
    {stdio: ['pipe', 'inherit', 'inherit']}
  );

  const write = (buf) => (ff.stdin.write(buf) ? Promise.resolve() : new Promise((r) => ff.stdin.once('drain', r)));

  for (let f = 0; f < total; f++) {
    await write(await grab(f / FPS));
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  cdp.ws.close();
  proc.kill();

  const info = execFileSync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration,size:stream=codec_name,width,height,avg_frame_rate',
    '-of', 'default=nw=1', outputMp4,
  ]).toString();
  console.log('\n' + outputMp4 + '\n' + info);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
