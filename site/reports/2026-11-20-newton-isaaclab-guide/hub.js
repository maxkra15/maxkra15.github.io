(() => {
  'use strict';
  const nav = document.querySelector('.tabs');
  const main = document.querySelector('main');
  const tabs = [...nav.querySelectorAll('[data-ch]')];
  const chapters = [...document.querySelectorAll('main > .chapter')];
  let current;

  function route() {
    let id = location.hash.slice(1);
    try {
      id = decodeURIComponent(id);
    } catch {
      id = '';
    }
    const target = document.getElementById(id);
    const chapter = target?.closest('.chapter')
      ?? chapters.find(el => el.dataset.ch === id)
      ?? chapters.find(el => el.dataset.ch === 'start');
    const previous = current;
    current = chapter.dataset.ch;

    chapters.forEach(el => { el.hidden = el !== chapter; });
    for (const tab of tabs) {
      if (tab.dataset.ch === current) tab.setAttribute('aria-current', 'page');
      else tab.removeAttribute('aria-current');
    }
    const active = tabs.find(tab => tab.dataset.ch === current);
    const bar = active.parentElement;
    const tabRect = active.getBoundingClientRect();
    const barRect = bar.getBoundingClientRect();
    if (tabRect.left < barRect.left) bar.scrollLeft += tabRect.left - barRect.left;
    if (tabRect.right > barRect.right) bar.scrollLeft += tabRect.right - barRect.right;

    for (let details = target?.closest('details'); details; details = details.parentElement.closest('details')) {
      details.open = true;
    }
    document.dispatchEvent(new CustomEvent('lessonchange', { detail: { current, previous } }));
    const anchor = target && target !== chapter && id !== current ? target : null;
    const focusTarget = anchor ?? chapter.querySelector('h1, h2');
    if (previous && focusTarget) {
      if (focusTarget.tabIndex < 0) focusTarget.tabIndex = -1;
      focusTarget.dataset.routeFocus = '';
      focusTarget.focus({ preventScroll: true });
    }
    if (anchor) anchor.scrollIntoView({ block: 'start' });
    else window.scrollTo({ top: 0 });
  }

  document.querySelector('.skip-link').addEventListener('click', event => {
    event.preventDefault();
    main.scrollIntoView({ block: 'start' });
    main.focus({ preventScroll: true });
  });

  // Other lesson scripts register listeners before the initial route.
  window.addEventListener('DOMContentLoaded', route, { once: true });
  window.addEventListener('hashchange', route);
})();
