// The two things Streamlit can't do by itself: the lens over the market map, and the function keys.
(() => {
  if (window.filingLens) return;  // Streamlit runs this again on every redraw; one set of listeners is enough
  window.filingLens = true;
  const SIZE = 190, ZOOM = 2.1;
  let lens, copy, board;

  document.addEventListener('mousemove', event => {
    const over = event.target.closest && event.target.closest('.board');
    if (!over || over.closest('.lens')) {
      if (lens) lens.style.display = 'none';
      return;
    }
    if (over !== board) {  // Streamlit may have redrawn the map, so the lens gets a fresh copy of it
      if (lens) lens.remove();
      board = over;
      lens = document.createElement('div');
      lens.className = 'lens';
      copy = over.cloneNode(true);
      copy.style.width = `${over.offsetWidth}px`;
      copy.setAttribute('aria-hidden', 'true');
      lens.append(copy);
      document.body.append(lens);
    }
    const area = over.getBoundingClientRect(), x = event.clientX - area.left, y = event.clientY - area.top;
    lens.style.display = 'block';
    lens.style.left = `${event.clientX - SIZE / 2}px`;
    lens.style.top = `${event.clientY - SIZE / 2}px`;
    copy.style.transform = `translate(${SIZE / 2 - x * ZOOM}px, ${SIZE / 2 - y * ZOOM}px) scale(${ZOOM})`;
  });

  document.addEventListener('keydown', event => {
    const index = {F1: 0, F2: 1, F3: 2}[event.key];
    if (index === undefined) return;
    event.preventDefault();
    document.querySelectorAll('[role="tab"]')[index]?.click();
  });
})();
