/* SIMPLE — project website interactions */
(function () {
  'use strict';

  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Scroll reveal + one-shot triggers (bars, counters) ---------- */
  const revealEls = document.querySelectorAll('.reveal');

  if ('IntersectionObserver' in window && !prefersReduced) {
    const io = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const el = entry.target;
          el.classList.add('is-visible');

          el.querySelectorAll('[data-count]').forEach(animateCount);

          obs.unobserve(el);
        });
      },
      { threshold: 0.18, rootMargin: '0px 0px -8% 0px' }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    // Fallback: show everything immediately
    revealEls.forEach((el) => el.classList.add('is-visible'));
    document.querySelectorAll('[data-count]').forEach((el) => {
      el.textContent = formatNum(+el.dataset.count) + (el.dataset.suffix || '');
    });
  }

  /* ---------- Animated counters ---------- */
  function animateCount(el) {
    const target = +el.dataset.count || 0;
    const suffix = el.dataset.suffix || '';
    const duration = 1400;
    const start = performance.now();

    function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
      el.textContent = formatNum(Math.round(target * eased)) + (p === 1 ? suffix : '');
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function formatNum(n) {
    return n.toLocaleString('en-US');
  }

  /* ---------- Theme toggle (light / dark) ---------- */
  const themeToggle = document.getElementById('themeToggle');
  const root = document.documentElement;
  if (themeToggle) {
    const setPressed = () =>
      themeToggle.setAttribute('aria-pressed', String(root.getAttribute('data-theme') === 'light'));
    setPressed();
    themeToggle.addEventListener('click', () => {
      const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
      setPressed();
    });
  }
  // Follow OS changes only when the user hasn't explicitly chosen.
  const media = window.matchMedia('(prefers-color-scheme: light)');
  const onSchemeChange = (e) => {
    try { if (localStorage.getItem('theme')) return; } catch (_) {}
    root.setAttribute('data-theme', e.matches ? 'light' : 'dark');
    if (themeToggle) themeToggle.setAttribute('aria-pressed', String(e.matches));
  };
  if (media.addEventListener) media.addEventListener('change', onSchemeChange);
  else if (media.addListener) media.addListener(onSchemeChange);

  /* ---------- Mobile nav toggle ---------- */
  const toggle = document.getElementById('navToggle');
  const links = document.getElementById('navLinks');
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
    });
    links.querySelectorAll('a').forEach((a) =>
      a.addEventListener('click', () => {
        links.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
      })
    );
  }

  /* ---------- Custom video controls ---------- */
  (function () {
    var video = document.getElementById('teaserVideo');
    if (!video) return;
    var playBtn     = document.getElementById('vctrPlay');
    var timeEl      = document.getElementById('vctrTime');
    var progressEl  = document.getElementById('vctrProgress');
    var fillEl      = document.getElementById('vctrFill');
    var thumbEl     = document.getElementById('vctrThumb');
    var muteBtn     = document.getElementById('vctrMute');
    var fsBtn       = document.getElementById('vctrFs');
    var wrap        = video.closest('.video-wrap');

    function fmt(s) {
      if (!isFinite(s)) return '0:00';
      var m = Math.floor(s / 60), sec = Math.floor(s % 60);
      return m + ':' + (sec < 10 ? '0' : '') + sec;
    }

    function syncProgress() {
      if (!video.duration) return;
      var pct = video.currentTime / video.duration * 100;
      fillEl.style.width  = pct + '%';
      thumbEl.style.left  = pct + '%';
      progressEl.setAttribute('aria-valuenow', Math.round(pct));
      timeEl.textContent  = fmt(video.currentTime) + ' / ' + fmt(video.duration);
    }

    video.addEventListener('timeupdate', syncProgress);
    video.addEventListener('loadedmetadata', function () {
      timeEl.textContent = '0:00 / ' + fmt(video.duration);
    });

    // Play / pause
    function togglePlay() {
      if (video.paused) video.play().catch(function () {});
      else video.pause();
    }
    video.addEventListener('click', togglePlay);
    playBtn.addEventListener('click', togglePlay);
    video.addEventListener('play',  function () { playBtn.classList.add('is-playing');    playBtn.setAttribute('aria-label', 'Pause'); });
    video.addEventListener('pause', function () { playBtn.classList.remove('is-playing'); playBtn.setAttribute('aria-label', 'Play');  });
    video.addEventListener('ended', function () { playBtn.classList.remove('is-playing'); playBtn.setAttribute('aria-label', 'Play');  });

    // Seeking — mouse and touch, with document-level drag tracking
    function applySeek(clientX) {
      var rect = progressEl.getBoundingClientRect();
      var pct  = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      if (video.duration) video.currentTime = pct * video.duration;
      fillEl.style.width = (pct * 100) + '%';
      thumbEl.style.left = (pct * 100) + '%';
    }
    var seeking = false;
    progressEl.addEventListener('mousedown', function (e) { seeking = true; applySeek(e.clientX); e.preventDefault(); });
    document.addEventListener('mousemove',  function (e) { if (seeking) applySeek(e.clientX); });
    document.addEventListener('mouseup',    function ()  { seeking = false; });
    progressEl.addEventListener('touchstart', function (e) { seeking = true; applySeek(e.touches[0].clientX); e.preventDefault(); }, { passive: false });
    document.addEventListener('touchmove',  function (e) { if (seeking) { applySeek(e.touches[0].clientX); e.preventDefault(); } }, { passive: false });
    document.addEventListener('touchend',   function ()  { seeking = false; });
    progressEl.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight') video.currentTime = Math.min(video.duration || 0, video.currentTime + 5);
      if (e.key === 'ArrowLeft')  video.currentTime = Math.max(0, video.currentTime - 5);
    });

    // Mute
    muteBtn.addEventListener('click', function () {
      video.muted = !video.muted;
    });
    video.addEventListener('volumechange', function () {
      muteBtn.classList.toggle('is-muted', video.muted);
      muteBtn.setAttribute('aria-label', video.muted ? 'Unmute' : 'Mute');
    });

    // Fullscreen
    fsBtn.addEventListener('click', function () {
      if (!document.fullscreenElement) {
        var req = wrap.requestFullscreen || wrap.webkitRequestFullscreen || wrap.mozRequestFullScreen;
        if (req) req.call(wrap);
      } else {
        var exit = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen;
        if (exit) exit.call(document);
      }
    });
    document.addEventListener('fullscreenchange', function () {
      var isFs = !!document.fullscreenElement;
      fsBtn.classList.toggle('is-fs', isFs);
      fsBtn.setAttribute('aria-label', isFs ? 'Exit fullscreen' : 'Fullscreen');
    });
  }());

  /* ---------- GitHub stars ---------- */
  fetch('https://api.github.com/repos/physical-superintelligence-lab/SIMPLE')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (data) {
      if (!data) return;
      var el = document.getElementById('ghStarCount');
      if (!el) return;
      var n = data.stargazers_count;
      el.textContent = n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
    })
    .catch(function () {});

  /* ---------- Copy BibTeX ---------- */
  document.querySelectorAll('.copy-btn').forEach(function (btn) {
    var code = btn.closest('.code').querySelector('code');
    if (!code) return;
    btn.addEventListener('click', async function () {
      try {
        await navigator.clipboard.writeText(code.innerText);
      } catch (e) {
        var r = document.createRange();
        r.selectNode(code);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
        try { document.execCommand('copy'); } catch (_) {}
        sel.removeAllRanges();
      }
      btn.textContent = 'Copied';
      btn.classList.add('copied');
      setTimeout(function () {
        btn.textContent = 'Copy';
        btn.classList.remove('copied');
      }, 1800);
    });
  });
})();
