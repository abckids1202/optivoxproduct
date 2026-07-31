
(function () {
  const toggle = document.getElementById('navToggle');
  const links = document.getElementById('navLinks');
  if (toggle && links) toggle.addEventListener('click', () => links.classList.toggle('open'));

  const obs = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (!entry.isIntersecting) return;
      entry.target.style.transitionDelay = `${Math.min(i, 5) * 50}ms`;
      entry.target.classList.add('in');
      obs.unobserve(entry.target);
    });
  }, { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach((el) => obs.observe(el));

  document.querySelectorAll('.faq-q').forEach((btn) => {
    btn.addEventListener('click', () => {
      const item = btn.closest('.faq-item');
      const open = item.classList.contains('open');
      document.querySelectorAll('.faq-item.open').forEach((el) => el.classList.remove('open'));
      if (!open) item.classList.add('open');
    });
  });
})();