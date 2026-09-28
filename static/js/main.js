// CollisionSplatting project page: math rendering, the n_sigma explorer, lazy video playback, BibTeX copy.

document.addEventListener("DOMContentLoaded", () => {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
    });
  }
  setupExplorer();
  setupVideos();
  setupCopy();
});

// ---------------------------------------------------------------- n_sigma explorer
// One splat N(0, Sigma) with principal std-devs (sx, sy) rotated by theta, and a sphere robot of radius a.
// For a robot centre c = d u (unit u): s^2 = d^2 / a^2, grad = 2 d u / a^2, sigma = (2 d / a^2) sqrt(u^T Sigma u).
// s~^2 = 1  <=>  d^2 - 2 n k d - a^2 = 0,  k = sqrt(u^T Sigma u)  =>  d = n k + sqrt(n^2 k^2 + a^2)   (exact boundary).
function setupExplorer() {
  const svg = document.querySelector("#explorer svg");
  if (!svg) return;
  const sx = 0.6, sy = 0.2, theta = (-25 * Math.PI) / 180, a = 0.4;
  const c = Math.cos(theta), s = Math.sin(theta);
  // Sigma = R diag(sx^2, sy^2) R^T  (SVG y points down, so negate the angle for a counter-clockwise tilt on screen)
  const Sxx = c * c * sx * sx + s * s * sy * sy, Syy = s * s * sx * sx + c * c * sy * sy, Sxy = c * s * (sx * sx - sy * sy);
  document.getElementById("splat").setAttribute("transform", `rotate(${(theta * 180) / Math.PI})`);
  const slider = document.getElementById("nsigma"), out = document.getElementById("nsigma-out");
  const keepout = document.getElementById("keepout"), robot = document.getElementById("robot"), dot = document.getElementById("robot-center");
  const readout = document.getElementById("readout");
  let pos = { x: 1.45, y: 0.75 };

  const kdir = (ux, uy) => Math.sqrt(ux * ux * Sxx + 2 * ux * uy * Sxy + uy * uy * Syy);
  function draw() {
    const n = parseFloat(slider.value);
    out.textContent = n.toFixed(1);
    let d = "";
    for (let i = 0; i <= 180; i++) {
      const t = (i / 180) * 2 * Math.PI, ux = Math.cos(t), uy = Math.sin(t), k = kdir(ux, uy);
      const r = n * k + Math.sqrt(n * n * k * k + a * a);
      d += (i ? "L" : "M") + (r * ux).toFixed(4) + " " + (r * uy).toFixed(4);
    }
    keepout.setAttribute("d", d + "Z");
    robot.setAttribute("cx", pos.x); robot.setAttribute("cy", pos.y);
    dot.setAttribute("cx", pos.x); dot.setAttribute("cy", pos.y);
    const dist = Math.hypot(pos.x, pos.y) || 1e-9, ux = pos.x / dist, uy = pos.y / dist;
    const s2 = (dist * dist) / (a * a), sigma = ((2 * dist) / (a * a)) * kdir(ux, uy);
    const st = Math.max(0, s2 - n * sigma), hit = st < 1;
    robot.classList.toggle("hit", hit);
    readout.innerHTML = `s² = ${s2.toFixed(2)} &nbsp; σ = ${sigma.toFixed(2)} &nbsp; s̃² = ${st.toFixed(2)}<br>` +
      (hit ? "<b>In collision</b> (s̃² &lt; 1)" : '<span class="free">Collision-free</span> (s̃² ≥ 1)');
  }

  function toSvg(evt) {
    const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY;
    const q = p.matrixTransform(svg.getScreenCTM().inverse());
    return { x: Math.max(-2.8, Math.min(2.8, q.x)), y: Math.max(-1.8, Math.min(1.8, q.y)) };
  }
  let dragging = false;
  robot.addEventListener("pointerdown", (e) => { dragging = true; robot.setPointerCapture(e.pointerId); });
  robot.addEventListener("pointermove", (e) => { if (dragging) { pos = toSvg(e); draw(); } });
  robot.addEventListener("pointerup", () => { dragging = false; });
  svg.addEventListener("pointerdown", (e) => { if (e.target !== robot) { pos = toSvg(e); draw(); } });
  robot.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 0.2 : 0.05, moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    if (!moves[e.key]) return;
    e.preventDefault(); pos = { x: pos.x + moves[e.key][0], y: pos.y + moves[e.key][1] }; draw();
  });
  slider.addEventListener("input", draw);
  draw();
}

// ---------------------------------------------------------------- videos: play only while visible, respect reduced motion
function setupVideos() {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const videos = document.querySelectorAll("video");
  if (reduce) {
    videos.forEach((v) => { v.removeAttribute("autoplay"); v.pause(); v.controls = true; });
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach(({ target, isIntersecting }) => {
      if (isIntersecting) { target.preload = "auto"; target.play().catch(() => { target.controls = true; }); }
      else target.pause();
    });
  }, { threshold: 0.25 });
  videos.forEach((v) => io.observe(v));
}

// ---------------------------------------------------------------- copy BibTeX
function setupCopy() {
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = document.querySelector(btn.dataset.copy).innerText;
      try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
      catch { btn.textContent = "Select and copy the text"; }
      setTimeout(() => (btn.textContent = "Copy BibTeX"), 2000);
    });
  });
}
