/* Rafeeq web UI: small enhancements only. All routing and agent logic stays on the server. */
(function () {
  'use strict';

  var body = document.body;
  var photos = {};          // file_id -> object URL of the photo the student picked
  var latestPhoto = null;   // object URL of the most recently picked photo

  function toast(message) {
    var el = document.getElementById('toast');
    if (!el) return;
    el.textContent = message;
    el.classList.add('show');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(function () { el.classList.remove('show'); }, 3200);
  }

  // Copy buttons ---------------------------------------------------------------
  document.addEventListener('click', function (event) {
    var btn = event.target.closest('[data-copy]');
    if (!btn) return;
    var text = btn.getAttribute('data-copy') || '';
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { toast(body.dataset.msgCopied); },
        function () { toast(body.dataset.msgCopyFailed); });
    } else {
      toast(body.dataset.msgCopyFailed);
    }
  });

  // File inputs: show the chosen name, keep the photo for the overlay ------------
  document.addEventListener('change', function (event) {
    var el = event.target;
    if (el.matches('[data-photo-input]')) {
      var label = el.closest('form').querySelector('[data-photo-name]');
      if (el.files && el.files[0]) {
        if (label) label.textContent = el.files[0].name;
        latestPhoto = URL.createObjectURL(el.files[0]);
      }
    }
    if (el.matches('[data-file-input]')) {
      var out = el.closest('form').querySelector('[data-file-name]');
      if (out) out.textContent = el.files && el.files[0] ? el.files[0].name : body.dataset.msgNoFile;
    }
    if (el.matches('[data-role-form] input[name="role"]')) {
      var studentOnly = el.closest('form').querySelector('[data-student-only]');
      if (studentOnly) studentOnly.hidden = el.value !== 'student';
    }
    if (el.matches('[data-toggles]')) {
      var target = document.querySelector(el.getAttribute('data-toggles'));
      if (target) target.hidden = !el.checked;
    }
  });

  // Enter sends, Shift+Enter adds a line --------------------------------------
  document.addEventListener('keydown', function (event) {
    var el = event.target;
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && el.matches('[data-submit-on-enter]')) {
      event.preventDefault();
      if (el.value.trim()) el.form.requestSubmit();
    }
  });

  // Photo overlay: show the student's photo and the boxes M3 returned -----------
  function fillPhotoSlots(root) {
    (root || document).querySelectorAll('[data-photo-slot]').forEach(function (slot) {
      if (slot.querySelector('img')) return;
      var id = slot.getAttribute('data-file-id');
      if (slot.hasAttribute('data-claim-latest') && latestPhoto && !photos[id]) photos[id] = latestPhoto;
      var src = photos[id];
      if (!src) return;
      var img = document.createElement('img');
      img.alt = '';
      img.onload = function () {
        var log = slot.closest('#chat-log');
        if (log) scrollChat(log);
      };
      img.src = src;
      slot.appendChild(img);
      var boxes = [];
      try { boxes = JSON.parse(slot.getAttribute('data-boxes') || '[]'); } catch (e) { boxes = []; }
      boxes.forEach(function (b) {
        var d = document.createElement('span');
        d.className = 'photo-box' + (b.status ? ' tq-box-' + b.status : '');
        // Box fractions are measured from the image's top-left corner, so use physical sides.
        d.style.left = (b.x * 100) + '%';
        d.style.top = (b.y * 100) + '%';
        d.style.width = (b.width * 100) + '%';
        d.style.height = (b.height * 100) + '%';
        slot.appendChild(d);
      });
    });
  }

  // Show the start of the newest reply (with the question just above it), not the bottom.
  function scrollChat(log) {
    var rows = log.querySelectorAll('.chat-row');
    var last = rows[rows.length - 1];
    if (!last) return;
    var prev = last.previousElementSibling;
    var anchor = prev && prev.classList.contains('from-user') ? prev : last;
    log.scrollTop = Math.max(0, anchor.offsetTop - 12);
  }

  body.addEventListener('htmx:afterSwap', function (event) {
    var target = event.detail.target;
    fillPhotoSlots(target);
    if (target && target.id === 'chat-log') {
      var empty = document.getElementById('chat-empty');
      if (empty) empty.remove();
      scrollChat(target);
      return;
    }
    var heading = target && target.querySelector ? target.querySelector('[data-focus-heading]') : null;
    if (heading) heading.focus({ preventScroll: false });
  });

  body.addEventListener('htmx:responseError', function () { toast(body.dataset.msgError); });
  body.addEventListener('htmx:sendError', function () { toast(body.dataset.msgError); });

  // If the HTMX script could not load, submit hx-post forms with fetch instead --
  window.addEventListener('load', function () {
    if (window.htmx) return;
    var csrf = (body.getAttribute('hx-headers') || '').match(/"X-CSRF-Token":\s*"([^"]+)"/);
    csrf = csrf ? csrf[1] : '';

    function send(el, url, data) {
      var target = document.querySelector(el.getAttribute('hx-target') || 'body');
      var swap = el.getAttribute('hx-swap') || 'innerHTML';
      var ind = document.querySelector(el.getAttribute('hx-indicator') || '#none');
      if (ind) ind.classList.add('htmx-request');
      fetch(url, { method: 'POST', body: data, credentials: 'same-origin', headers: { 'HX-Request': 'true', 'X-CSRF-Token': csrf } })
        .then(function (r) {
          var redirect = r.headers.get('HX-Redirect');
          if (redirect) { window.location = redirect; return ''; }
          if (!r.ok) throw new Error(r.status);
          return r.text();
        })
        .then(function (html) {
          if (!html || !target) return;
          if (swap === 'beforeend') target.insertAdjacentHTML('beforeend', html);
          else if (swap === 'outerHTML') target.outerHTML = html;
          else target.innerHTML = html;
          body.dispatchEvent(new CustomEvent('htmx:afterSwap', { detail: { target: target } }));
          if (el.tagName === 'FORM' && (el.getAttribute('hx-on::after-request') || '').indexOf('reset') >= 0) el.reset();
        })
        .catch(function () { toast(body.dataset.msgError); })
        .finally(function () { if (ind) ind.classList.remove('htmx-request'); });
    }

    document.addEventListener('submit', function (event) {
      var form = event.target.closest('form[hx-post]');
      if (!form) return;
      event.preventDefault();
      var data = new FormData(form, event.submitter || undefined);
      send(form, form.getAttribute('hx-post'), data);
    });
    document.addEventListener('click', function (event) {
      var btn = event.target.closest('button[hx-post]:not([type="submit"])');
      if (!btn) return;
      var data = new FormData();
      try {
        var vals = JSON.parse(btn.getAttribute('hx-vals') || '{}');
        Object.keys(vals).forEach(function (k) { data.append(k, vals[k]); });
      } catch (e) { /* ignore */ }
      send(btn, btn.getAttribute('hx-post'), data);
    });
  });
}());
