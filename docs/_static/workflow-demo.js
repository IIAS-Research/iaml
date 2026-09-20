/* GSAP pins the recorded walkthrough. Reduced motion keeps the complete page readable. */
(() => {
  "use strict";

  function enhance(root, gsap, ScrollTrigger) {
    const pin = root.querySelector(".iaml-story-pin");
    const progress = root.querySelector(".iaml-story-progress");
    const buttons = [...root.querySelectorAll(".iaml-story-controls [data-story-step]")];
    const phases = buttons.map(button => root.querySelector(
      '[data-story-phase="' + button.dataset.storyStep + '"]'));
    const captions = buttons.map(button => root.querySelector(
      '[data-story-caption="' + button.dataset.storyStep + '"]'));
    if (!pin || !progress || buttons.length < 2
        || phases.some(phase => !phase) || captions.some(caption => !caption)) return;

    let timeline = null;
    let active = -1;
    const last = phases.length - 1;

    function syncAccessibility() {
      if (!timeline) return;
      // Switch descriptions when the incoming visual becomes dominant.
      const next = Math.min(last, Math.floor(timeline.time() + .2));
      if (next !== active) {
        const moveFocus = active >= 0 && phases[active].contains(document.activeElement);
        active = next;
        root.dataset.active = String(active);
        phases.forEach((phase, i) => {
          phase.inert = i !== active;
          phase.setAttribute("aria-hidden", String(i !== active));
          captions[i].hidden = i !== active;
          if (i === active) buttons[i].setAttribute("aria-current", "step");
          else buttons[i].removeAttribute("aria-current");
        });
        if (moveFocus) buttons[active].focus({ preventScroll: true });
      }
      const percent = Math.round(timeline.progress() * 100);
      progress.setAttribute("aria-valuenow", String(percent));
      progress.setAttribute("aria-valuetext",
        "Reading progress: " + percent + "% — " + buttons[active].textContent.trim());
    }

    root.querySelectorAll("[data-story-step]").forEach(button => {
      button.addEventListener("click", event => {
        if (!timeline || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        window.scrollTo({
          top: timeline.scrollTrigger.labelToScroll("step-" + button.dataset.storyStep),
          behavior: "smooth",
        });
      });
    });

    root.classList.add("is-static-story");
    gsap.matchMedia().add(
      "(min-height: 560px) and (prefers-reduced-motion: no-preference)", () => {
        root.classList.remove("is-static-story");
        root.classList.add("is-scroll-story");
        active = -1;
        gsap.set(root, { "--progress": 0, "--story": 0 });
        gsap.set(phases, { "--in": 0, "--out": 0, "--visibility": 0 });
        gsap.set(phases[0], { "--in": 1, "--visibility": 1 });

        timeline = gsap.timeline({
          defaults: { ease: "none", immediateRender: false },
          onUpdate: syncAccessibility,
          scrollTrigger: {
            trigger: root,
            pin,
            start: () => "clamp(top " + (parseFloat(window.getComputedStyle(root)
              .getPropertyValue("--pin-top")) || 0) + "px)",
            end: () => "+=" + window.innerHeight * last * .85,
            scrub: .45,
            snap: {
              snapTo: "labels",
              directional: false,
              inertia: false,
              delay: .15,
              duration: { min: .25, max: .45 },
              ease: "power1.inOut",
            },
            anticipatePin: 1,
            invalidateOnRefresh: true,
          },
        });
        // Explicit starts remain deterministic after refresh invalidates cached tween values.
        timeline.fromTo(root, { "--progress": 0, "--story": 0 },
          { "--progress": 1, "--story": last, duration: last }, 0);
        phases.forEach((phase, i) => {
          timeline.addLabel("step-" + i, i);
          if (i > 0) {
            timeline.fromTo(phase, { "--in": 0, "--visibility": 0 },
              { "--in": 1, "--visibility": 1, duration: .45 }, i - .5);
          }
          if (i < last) {
            timeline.fromTo(phase, { "--out": 0, "--visibility": 1 },
              { "--out": 1, "--visibility": 0, duration: .30 }, i + .65);
          }
        });
        timeline.scrollTrigger.refresh();
        syncAccessibility();

        // matchMedia reverts GSAP styles and pinning. Restore the native document state.
        return () => {
          timeline = null;
          root.classList.remove("is-scroll-story");
          root.classList.add("is-static-story");
          delete root.dataset.active;
          phases.forEach((phase, i) => {
            phase.inert = false;
            phase.removeAttribute("aria-hidden");
            captions[i].hidden = false;
            buttons[i].removeAttribute("aria-current");
          });
        };
      });
  }

  function initialize() {
    const { gsap, ScrollTrigger } = window;
    if (!gsap || !ScrollTrigger) return;
    gsap.registerPlugin(ScrollTrigger);
    document.querySelectorAll("[data-iaml-demo]").forEach(root => enhance(root, gsap, ScrollTrigger));
    if (document.fonts) document.fonts.ready.then(() => ScrollTrigger.refresh());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
