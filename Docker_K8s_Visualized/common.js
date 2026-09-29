/* Shared script for the Docker & Kubernetes pages: top nav, prev/next pager, copy buttons, Simple/Detailed switch. */
const PAGES = [
  ['index.html', 'Overview'],
  ['00_install.html', 'Install Docker Desktop'],
  ['01_why_containers.html', 'Why containers?'],
  ['02_images_layers.html', 'Images & layers'],
  ['03_run_container.html', 'Running a container'],
  ['04_compose.html', 'Docker Compose'],
  ['05_why_kubernetes.html', 'Why Kubernetes?'],
  ['06_pod.html', 'Pod'],
  ['07_deployment.html', 'Deployment'],
  ['08_service.html', 'Service'],
  ['09_config_secret.html', 'ConfigMap, Secret, Namespace'],
  ['10_health_resources.html', 'Health checks & resources'],
  ['11_autoscaling.html', 'Autoscaling (HPA)'],
  ['12_lab_commands.html', 'Hands-on lab'],
];
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const sleep = ms => new Promise(r => setTimeout(r, ms));

(function () {
  const here = decodeURIComponent(location.pathname.split('/').pop() || 'index.html');
  const idx = PAGES.findIndex(p => p[0] === here);
  const nav = $('.nav');
  if (nav) nav.innerHTML = PAGES.map(([h, t], i) => `<a href="${h}" class="${h === here ? 'on' : ''}" title="${t}">${i === 0 ? 'Home' : i - 1}</a>`).join('');
  const pager = $('nav.pager[data-auto]');
  if (pager && idx >= 0) {
    const prev = PAGES[idx - 1], next = PAGES[idx + 1];
    pager.innerHTML = (prev ? `<a href="${prev[0]}"><small>← Previous</small>${idx - 1 > 0 ? (idx - 2) + ' · ' : ''}${prev[1]}</a>` : '<span></span>') +
      (next ? `<a class="next" href="${next[0]}"><small>Next →</small>${idx} · ${next[1]}</a>` : `<a class="next" href="index.html"><small>Done! →</small>Back to the overview</a>`);
  }
  // copy buttons on every command line
  $$('.term .cmd').forEach(c => {
    const b = document.createElement('button'); b.className = 'copy'; b.type = 'button'; b.textContent = 'copy';
    b.onclick = () => { const t = c.firstChild.textContent.trim();
      try { navigator.clipboard.writeText(t).then(() => { b.textContent = 'copied'; setTimeout(() => b.textContent = 'copy', 1200); }); } catch (e) { b.textContent = 'select it'; } };
    c.appendChild(b);
  });
  // Simple / Detailed switch (hides elements with class "adv"), remembered per browser
  const mb = $('#modeBtn');
  if (mb) {
    let m = 'simple'; try { m = localStorage.getItem('dk-mode') || 'simple'; } catch (e) { }
    const set = s => { document.body.classList.toggle('simple', s); mb.textContent = s ? '👶 Simple view (click for details)' : '🎓 Detailed view (click to simplify)'; };
    set(m === 'simple');
    mb.onclick = () => { const s = !document.body.classList.contains('simple'); set(s); try { localStorage.setItem('dk-mode', s ? 'simple' : 'detailed'); } catch (e) { } window.dispatchEvent(new Event('resize')); };
  }
  // step checklists remember their ticks
  $$('.stepcard > h3 input[type=checkbox]').forEach((cb, i) => {
    const key = 'dk-step-' + here + '-' + i, card = cb.closest('.stepcard');
    try { cb.checked = localStorage.getItem(key) === '1'; } catch (e) { }
    card.classList.toggle('done', cb.checked);
    cb.addEventListener('change', () => { card.classList.toggle('done', cb.checked); try { localStorage.setItem(key, cb.checked ? '1' : '0'); } catch (e) { } });
  });
})();
