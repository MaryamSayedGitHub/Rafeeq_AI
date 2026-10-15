/* The tutor agent's small enhancements: answer the 3 MCQ in the page, flip flashcards. The only server call is the quiz result (for the progress page). */
(function () {
  'use strict';

  // The finished quiz is a result for the student's progress (once per quiz): the server gives it to M4.
  function sendResult(quiz, correct, total) {
    if (quiz.dataset.sent || !quiz.dataset.topic || !quiz.dataset.resultUrl) return;
    quiz.dataset.sent = '1';
    var csrf = (document.body.getAttribute('hx-headers') || '').match(/"X-CSRF-Token":\s*"([^"]+)"/);
    var data = new URLSearchParams({ topic: quiz.dataset.topic, correct: correct, total: total });
    fetch(quiz.dataset.resultUrl, { method: 'POST', body: data, credentials: 'same-origin', headers: { 'X-CSRF-Token': csrf ? csrf[1] : '' } }).catch(function () {});
  }

  document.addEventListener('click', function (event) {
    var opt = event.target.closest('.tq-opt');
    if (opt) {
      var q = opt.closest('.tq-q');
      if (!q || q.dataset.done) return;
      q.dataset.done = '1';
      var right = parseInt(q.dataset.answer, 10);
      var picked = parseInt(opt.dataset.i, 10);
      var quiz = q.closest('.tq-quiz');
      var buttons = q.querySelectorAll('.tq-opt');
      buttons.forEach(function (b) { b.disabled = true; });
      if (buttons[right]) buttons[right].classList.add('tq-right');
      var fb = q.querySelector('.tq-fb');
      if (picked === right) {
        q.dataset.score = '1';
        fb.className = 'tq-fb tq-right';
        fb.textContent = quiz.dataset.msgOk + (q.dataset.why ? ' ' + q.dataset.why : '');
      } else {
        opt.classList.add('tq-wrong');
        fb.className = 'tq-fb tq-wrong';
        fb.textContent = quiz.dataset.msgBad + ' ' + (buttons[right] ? buttons[right].textContent : '') + (q.dataset.why ? '. ' + q.dataset.why : '');
      }
      var all = quiz.querySelectorAll('.tq-q');
      var done = quiz.querySelectorAll('.tq-q[data-done]');
      if (done.length === all.length) {
        var n = quiz.querySelectorAll('.tq-q[data-score]').length;
        quiz.querySelector('.tq-score').textContent = quiz.dataset.msgScore.replace('{n}', n).replace('{total}', all.length);
        sendResult(quiz, n, all.length);
      }
      return;
    }
    var card = event.target.closest('.tq-card');
    if (card) {
      var flipped = card.classList.toggle('tq-flipped');
      card.setAttribute('aria-pressed', flipped ? 'true' : 'false');
    }
  });
}());
