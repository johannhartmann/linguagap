// Bilingual transcript export — shared between host (index.html) and
// viewer (viewer.html). Exposes helpers on window.TranscriptExport.
//
// Usage:
//   TranscriptExport.download({
//       segments: <array of segment objects as sent by the server>,
//       foreignLang: 'tr',
//       source: 'host' | 'viewer',
//       langNames: { tr: 'Turkish', ... },   // optional, falls back to the code
//   });
(function (global) {
    'use strict';

    function formatTime(seconds) {
        if (seconds == null || !isFinite(seconds)) return '';
        const total = Math.max(0, Math.floor(seconds));
        const h = Math.floor(total / 3600);
        const m = Math.floor((total % 3600) / 60);
        const s = total % 60;
        const pad = (n) => String(n).padStart(2, '0');
        return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
    }

    function escapeHtml(text) {
        return String(text == null ? '' : text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function safeLangCode(lang) {
        return typeof lang === 'string' && /^[a-z]{2,3}$/.test(lang) ? lang : '';
    }

    function filename(foreignLang) {
        const d = new Date();
        const pad = (n) => String(n).padStart(2, '0');
        const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
        const safe = safeLangCode(foreignLang);
        const suffix = safe && safe !== 'de' ? `-de-${safe}` : '';
        return `gespraechsprotokoll-${stamp}${suffix}.html`;
    }

    function buildHtml(segments, foreignLang, source, langNames) {
        const names = langNames || {};
        const foreignLabel = foreignLang && names[foreignLang] ? names[foreignLang] : (foreignLang || 'Foreign');
        const safeForeignAttr = safeLangCode(foreignLang);
        const rows = (Array.isArray(segments) ? segments : [])
            .filter(seg => seg && seg.final)
            .map(seg => {
                const time = formatTime(seg.abs_start);
                const isGerman = seg.src_lang === 'de';
                const role = seg.speaker_role === 'german' || isGerman ? 'de' : 'fg';
                const speaker = role === 'de' ? 'Host' : 'Gast';
                const translations = seg.translations || {};
                const germanText = isGerman ? seg.src : (translations.de || '');
                const foreignText = isGerman
                    ? (foreignLang ? (translations[foreignLang] || '') : '')
                    : seg.src;
                return `    <tr class="${role}">
      <td class="t-time">${escapeHtml(time)}</td>
      <td class="t-speaker">${escapeHtml(speaker)}</td>
      <td class="t-foreign" lang="${escapeHtml(safeForeignAttr)}">${escapeHtml(foreignText)}</td>
      <td class="t-german" lang="de">${escapeHtml(germanText)}</td>
    </tr>`;
            })
            .join('\n');
        const now = new Date();
        const dateStr = now.toLocaleString('de-DE');
        return `<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Gesprächsprotokoll</title>
<style>
  body { font-family: 'Assistant', 'Helvetica Neue', Arial, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }
  h1 { color: #0099a8; border-bottom: 2px solid #0099a8; padding-bottom: 0.3em; }
  .meta { color: #555; margin-bottom: 1.5em; font-size: 0.9rem; line-height: 1.6; }
  table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  thead th { background: #0099a8; color: white; padding: 10px; text-align: left; font-weight: 600; font-size: 0.9rem; }
  tbody td { padding: 10px 12px; border-bottom: 1px solid #eaeaea; vertical-align: top; font-size: 0.95rem; line-height: 1.5; word-wrap: break-word; }
  tr.de td.t-speaker { color: #0099a8; font-weight: 600; }
  tr.fg td.t-speaker { color: #c46a00; font-weight: 600; }
  td.t-time { width: 80px; color: #999; font-size: 0.8rem; white-space: nowrap; }
  td.t-speaker { width: 70px; }
  td.t-foreign, td.t-german { width: calc(50% - 75px); }
  footer { margin-top: 2em; padding-top: 1em; border-top: 1px solid #eaeaea; color: #888; font-size: 0.8rem; }
  @media print { body { margin: 1cm; max-width: none; } thead { display: table-header-group; } tr { page-break-inside: avoid; } }
</style>
</head>
<body>
<h1>Gesprächsprotokoll / Conversation Transcript</h1>
<div class="meta">
  <div><strong>Erstellt / Created:</strong> ${escapeHtml(dateStr)}</div>
  <div><strong>Sprachen / Languages:</strong> Deutsch &harr; ${escapeHtml(foreignLabel)}</div>
  <div><strong>Quelle / Source:</strong> ${escapeHtml(source)}</div>
</div>
<table>
  <thead>
    <tr>
      <th>Zeit</th>
      <th>Sprecher</th>
      <th>${escapeHtml(foreignLabel)}</th>
      <th>Deutsch</th>
    </tr>
  </thead>
  <tbody>
${rows}
  </tbody>
</table>
<footer>Synia Gespr&auml;chsassistent (SGA) &mdash; automatisierte &Uuml;bersetzung, bitte wichtige Informationen gegenpr&uuml;fen.</footer>
</body>
</html>`;
    }

    // Triggers a browser download for the given transcript.
    // Throws on failure so callers can surface a user-visible error.
    function download(opts) {
        const segs = opts.segments || [];
        const foreignLang = opts.foreignLang || null;
        const source = opts.source || 'unknown';
        const langNames = opts.langNames || null;

        const html = buildHtml(segs, foreignLang, source, langNames);
        const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = filename(foreignLang);
            a.rel = 'noopener';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } finally {
            // Revoke after the browser has a chance to start the download.
            // a.click() is synchronous so the blob is captured before we get
            // here, but the safety net of a short delay avoids edge cases on
            // older mobile browsers.
            setTimeout(() => URL.revokeObjectURL(url), 1500);
        }
    }

    global.TranscriptExport = {
        buildHtml: buildHtml,
        filename: filename,
        download: download,
    };
})(typeof window !== 'undefined' ? window : globalThis);
