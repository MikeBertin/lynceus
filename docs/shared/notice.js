// Lynceus: hover/click "why" popovers for the "things to notice" cards.
// Each trigger is <span class="x why">…</span> with a sibling <div class="explain">…</div>.
(function () {
  "use strict";
  const pop = document.getElementById("pop");
  if (!pop) return;
  let pinned = null;

  function show(el) {
    const body = el.parentElement.querySelector(".explain");
    if (!body) return;
    pop.innerHTML = body.innerHTML;
    pop.classList.add("show");
    const r = el.getBoundingClientRect(), p = pop.getBoundingClientRect(), m = 10;
    const left = Math.max(m, Math.min(r.left + r.width / 2 - p.width / 2, innerWidth - p.width - m));
    let top = r.top - p.height - 10;
    if (top < m) top = r.bottom + 10;
    pop.style.left = left + "px"; pop.style.top = top + "px";
  }
  const hide = () => pop.classList.remove("show");

  document.querySelectorAll(".why").forEach((el) => {
    el.tabIndex = 0; el.setAttribute("role", "button");
    el.addEventListener("mouseenter", () => { if (!pinned) show(el); });
    el.addEventListener("mouseleave", () => { if (!pinned) hide(); });
    el.addEventListener("focus", () => { if (!pinned) show(el); });
    el.addEventListener("blur", () => { if (!pinned) hide(); });
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      if (pinned === el) { pinned = null; hide(); } else { pinned = el; show(el); }
    });
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); }
    });
  });
  pop.addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("click", () => { if (pinned) { pinned = null; hide(); } });
  addEventListener("keydown", (e) => { if (e.key === "Escape" && pinned) { pinned = null; hide(); } });
  addEventListener("scroll", () => { if (pinned) show(pinned); }, true);
})();
