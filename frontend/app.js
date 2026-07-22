// ---------------------------------------------------------------------
// Pride & Prejudice voice companion — client
//
// Talks directly to our own FastAPI server's /api/offer endpoint using
// plain WebRTC (Pipecat's SmallWebRTCTransport on the backend). No Daily.co,
// no third-party call infrastructure, no per-minute fee.
//
// The orb's size is driven purely by live audio amplitude:
//   - local mic level  -> orb grows while YOU talk
//   - remote TTS level -> orb grows/pulses while the AGENT talks
// This needs no custom signaling protocol from the backend at all.
// ---------------------------------------------------------------------

const el = {
  status: document.getElementById('status'),
  hint: document.getElementById('hint'),
  scene: document.querySelector('.scene'),
  orbWrap: document.querySelector('.orb-wrap'),
  orbCanvas: document.getElementById('orb'),
  micBtn: document.getElementById('mic-btn'),
  waveform: document.getElementById('waveform'),
};

const STATE = {
  IDLE: 'Tap the mic to start',
  CONNECTING: 'Connecting…',
  LISTENING: 'Listening…',
  THINKING: 'Thinking…',
  SPEAKING: 'Speaking…',
};

let pc = null;
let localStream = null;
let audioCtx = null;
let localAnalyser = null;
let remoteAnalyser = null;
let rafId = null;
let callActive = false;
let connecting = false; // guards against a second startCall() firing before the first finishes

// Rolling buffer of recent local-mic amplitude samples, used to draw the
// scrolling waveform. New samples are pushed on the right and the whole
// thing visually drifts leftward as time passes.
const WAVE_SAMPLES = 120;
const waveBuffer = new Array(WAVE_SAMPLES).fill(0);

// ---------------------------------------------------------------- orb ----

const orbCtx = el.orbCanvas.getContext('2d');
let orbLevel = 0; // smoothed 0..1, whichever of local/remote is louder
let orbPhase = 0;
let orbSpeakerIsAgent = false;

function drawOrb() {
  const w = el.orbCanvas.width;
  const h = el.orbCanvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const baseR = w * 0.30;

  orbPhase += 0.012;

  const pulse = Math.sin(orbPhase) * 0.02;
  const scale = 1 + orbLevel * 0.38 + pulse;
  const r = baseR * scale;

  orbCtx.clearRect(0, 0, w, h);

  // Soft outer glow
  const glow = orbCtx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r * 1.9);
  const glowColor = orbSpeakerIsAgent ? '109,77,255' : '155,139,255';
  glow.addColorStop(0, `rgba(${glowColor},${0.35 + orbLevel * 0.25})`);
  glow.addColorStop(1, 'rgba(10,6,25,0)');
  orbCtx.fillStyle = glow;
  orbCtx.beginPath();
  orbCtx.arc(cx, cy, r * 1.9, 0, Math.PI * 2);
  orbCtx.fill();

  // Layered swirling body (a few offset, semi-transparent blobs give the
  // "glassy marbled sphere" look from the reference art without needing
  // an actual 3D renderer)
  const layers = 5;
  for (let i = 0; i < layers; i++) {
    const t = orbPhase * (0.6 + i * 0.15) + i * 1.7;
    const ox = Math.cos(t) * r * 0.14;
    const oy = Math.sin(t * 1.3) * r * 0.14;
    const lr = r * (0.72 + 0.06 * Math.sin(t * 0.7 + i));

    const grad = orbCtx.createRadialGradient(
      cx + ox - lr * 0.25, cy + oy - lr * 0.3, lr * 0.05,
      cx + ox, cy + oy, lr
    );
    const hue = 258 - i * 6;
    grad.addColorStop(0, `hsla(${hue}, 90%, ${72 - i * 4}%, 0.85)`);
    grad.addColorStop(0.55, `hsla(${hue}, 80%, ${38 - i * 3}%, 0.75)`);
    grad.addColorStop(1, `hsla(${hue + 8}, 70%, 8%, 0.0)`);

    orbCtx.fillStyle = grad;
    orbCtx.beginPath();
    orbCtx.arc(cx + ox, cy + oy, lr, 0, Math.PI * 2);
    orbCtx.fill();
  }

  // Bright core highlight
  const core = orbCtx.createRadialGradient(
    cx - r * 0.28, cy - r * 0.32, 1, cx, cy, r * 0.9
  );
  core.addColorStop(0, 'rgba(230,225,255,0.55)');
  core.addColorStop(0.4, 'rgba(155,139,255,0.12)');
  core.addColorStop(1, 'rgba(155,139,255,0)');
  orbCtx.fillStyle = core;
  orbCtx.beginPath();
  orbCtx.arc(cx, cy, r * 0.95, 0, Math.PI * 2);
  orbCtx.fill();
}

// ------------------------------------------------------------ waveform ---

const waveCtx = el.waveform.getContext('2d');

function drawWaveform() {
  const w = el.waveform.width;
  const h = el.waveform.height;
  waveCtx.clearRect(0, 0, w, h);

  const slot = w / WAVE_SAMPLES;
  const midY = h / 2;

  waveCtx.lineWidth = 3;
  waveCtx.lineCap = 'round';

  for (let i = 0; i < WAVE_SAMPLES; i++) {
    // index 0 = oldest (leftmost, faded), last = newest (rightmost, bright)
    const amp = waveBuffer[i];
    const barH = Math.max(3, amp * h * 0.85);
    const x = i * slot + slot / 2;
    const alpha = 0.15 + 0.85 * (i / WAVE_SAMPLES);

    waveCtx.strokeStyle = `rgba(155, 139, 255, ${alpha})`;
    waveCtx.beginPath();
    waveCtx.moveTo(x, midY - barH / 2);
    waveCtx.lineTo(x, midY + barH / 2);
    waveCtx.stroke();
  }
}

function pushWaveSample(amp) {
  waveBuffer.shift(); // drop oldest (left edge)
  waveBuffer.push(amp); // new sample enters on the right, drifts left over time
}

// ------------------------------------------------------------- levels ----

function getRmsLevel(analyser, buffer) {
  analyser.getByteTimeDomainData(buffer);
  let sumSquares = 0;
  for (let i = 0; i < buffer.length; i++) {
    const norm = (buffer[i] - 128) / 128;
    sumSquares += norm * norm;
  }
  return Math.sqrt(sumSquares / buffer.length); // 0..~1
}

function animationLoop() {
  let localLevel = 0;
  let remoteLevel = 0;

  if (localAnalyser) {
    const buf = new Uint8Array(localAnalyser.fftSize);
    localLevel = Math.min(1, getRmsLevel(localAnalyser, buf) * 4.5);
  }
  if (remoteAnalyser) {
    const buf = new Uint8Array(remoteAnalyser.fftSize);
    remoteLevel = Math.min(1, getRmsLevel(remoteAnalyser, buf) * 4.5);
  }

  // Orb reacts to whichever side is currently louder.
  const targetLevel = Math.max(localLevel, remoteLevel);
  orbLevel += (targetLevel - orbLevel) * 0.25; // smoothing
  orbSpeakerIsAgent = remoteLevel > localLevel;
  drawOrb();

  if (callActive) {
    pushWaveSample(localLevel);
    drawWaveform();

    if (remoteLevel > 0.06) {
      setStatus(STATE.SPEAKING);
    } else if (localLevel > 0.06) {
      setStatus(STATE.LISTENING);
    } else {
      setStatus(STATE.THINKING);
    }
  }

  rafId = requestAnimationFrame(animationLoop);
}

// --------------------------------------------------------------- ui ------

function setStatus(text) {
  if (el.status.textContent !== text) el.status.textContent = text;
}

function setCallUi(active) {
  callActive = active;
  el.scene.classList.toggle('active', active);
  el.micBtn.classList.toggle('hidden', active);
  el.micBtn.disabled = active; // re-enabled automatically once hidden state clears on hangup
  el.waveform.classList.toggle('hidden', !active);
  el.orbWrap.classList.toggle('listening', active);
  if (!active) {
    waveBuffer.fill(0);
    setStatus(STATE.IDLE);
  }
}

// ------------------------------------------------------------ webrtc -----

async function startCall() {
  if (connecting || callActive) return; // already connecting or already live — ignore extra clicks
  connecting = true;
  try {
    setStatus(STATE.CONNECTING);
    localStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const localSource = audioCtx.createMediaStreamSource(localStream);
    localAnalyser = audioCtx.createAnalyser();
    localAnalyser.fftSize = 512;
    localSource.connect(localAnalyser);

    pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    });

    localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

    const remoteAudioEl = new Audio();
    remoteAudioEl.autoplay = true;

    pc.ontrack = (event) => {
      remoteAudioEl.srcObject = event.streams[0];
      const remoteSource = audioCtx.createMediaStreamSource(event.streams[0]);
      remoteAnalyser = audioCtx.createAnalyser();
      remoteAnalyser.fftSize = 512;
      remoteSource.connect(remoteAnalyser);
    };

    pc.onconnectionstatechange = () => {
      if (['disconnected', 'failed', 'closed'].includes(pc.connectionState)) {
        endCall();
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIceGatheringComplete(pc);

    const response = await fetch('/api/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      }),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Signaling failed: ${response.status}`);
    }
    const answer = await response.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));

    setCallUi(true);
    setStatus(STATE.LISTENING);
    connecting = false;
  } catch (err) {
    console.error('Failed to start call:', err);
    setStatus(err.message || 'Could not connect — check mic permissions');
    setTimeout(() => setStatus(STATE.IDLE), 4000);
    cleanupConnection();
    connecting = false;
    el.micBtn.disabled = false;
  }
}

function waitForIceGatheringComplete(peerConnection) {
  if (peerConnection.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    function check() {
      if (peerConnection.iceGatheringState === 'complete') {
        peerConnection.removeEventListener('icegatheringstatechange', check);
        resolve();
      }
    }
    peerConnection.addEventListener('icegatheringstatechange', check);
  });
}

function cleanupConnection() {
  if (localStream) {
    localStream.getTracks().forEach((t) => t.stop());
    localStream = null;
  }
  if (pc) {
    pc.close();
    pc = null;
  }
  localAnalyser = null;
  remoteAnalyser = null;
}

function endCall() {
  cleanupConnection();
  setCallUi(false);
}

// ------------------------------------------------------------ events -----

el.micBtn.addEventListener('click', () => {
  if (connecting || callActive) return;
  el.micBtn.disabled = true;
  startCall();
});

// Click the orb itself to hang up mid-call.
el.orbCanvas.addEventListener('click', () => {
  if (callActive) endCall();
});

setStatus(STATE.IDLE);
rafId = requestAnimationFrame(animationLoop);
