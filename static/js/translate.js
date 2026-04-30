// One-shot text translation page — POSTs to /api/translate and renders
// the result. Pulls the language list from /api/languages so it stays in
// sync with the server's authoritative registry.

(() => {
    /**
     * @param {HTMLSelectElement} selectEl
     * @param {Array<{ code: string; label: string }>} options
     * @param {string} defaultCode
     */
    function fillLangSelect(selectEl, options, defaultCode) {
        for (const { code, label } of options) {
            const opt = document.createElement('option');
            opt.value = code;
            opt.textContent = label;
            if (code === defaultCode) opt.selected = true;
            selectEl.appendChild(opt);
        }
    }

    const srcLang = /** @type {HTMLSelectElement} */ (document.getElementById('srcLang'));
    const tgtLang = /** @type {HTMLSelectElement} */ (document.getElementById('tgtLang'));
    const sourceText = /** @type {HTMLTextAreaElement} */ (document.getElementById('sourceText'));
    const charCounter = /** @type {HTMLElement} */ (document.getElementById('charCounter'));
    const translateBtn = /** @type {HTMLButtonElement} */ (document.getElementById('translateBtn'));
    const translateSpinner = /** @type {HTMLElement} */ (
        document.getElementById('translateSpinner')
    );
    const translateLabel = /** @type {HTMLElement} */ (document.getElementById('translateLabel'));
    const printBtn = /** @type {HTMLButtonElement} */ (document.getElementById('printBtn'));
    const result = /** @type {HTMLElement} */ (document.getElementById('result'));
    const errorBanner = /** @type {HTMLElement} */ (document.getElementById('errorBanner'));

    /** @param {string} msg */
    function showError(msg) {
        errorBanner.textContent = msg;
        errorBanner.classList.add('visible');
    }
    function clearError() {
        errorBanner.classList.remove('visible');
    }

    function updateCounter() {
        charCounter.textContent = `Zeichen: ${sourceText.value.length}`;
    }
    sourceText.addEventListener('input', updateCounter);
    updateCounter();

    // Populate dropdowns from the server's language registry. The submit
    // button stays disabled until both lists are filled.
    (async () => {
        const resp = await fetch('/api/languages?scope=translate');
        if (!resp.ok) {
            showError('Sprachliste konnte nicht geladen werden.');
            return;
        }
        const options = await resp.json();
        fillLangSelect(srcLang, options, 'de');
        fillLangSelect(tgtLang, options, 'en');
    })();

    translateBtn.addEventListener('click', async () => {
        const text = sourceText.value.trim();
        clearError();
        if (!text) {
            showError('Bitte einen Text eingeben.');
            return;
        }
        if (srcLang.value === tgtLang.value) {
            showError('Quell- und Zielsprache sind identisch.');
            return;
        }
        translateBtn.disabled = true;
        translateSpinner.style.display = 'inline-block';
        translateLabel.textContent = 'Übersetze…';
        try {
            const resp = await fetch('/api/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    src_lang: srcLang.value,
                    tgt_lang: tgtLang.value,
                }),
            });
            if (resp.status === 401) {
                window.location.href = '/login';
                return;
            }
            if (!resp.ok) {
                let detail = '';
                try {
                    detail = (await resp.json()).detail || '';
                } catch {
                    /* ignore */
                }
                showError(`Übersetzung fehlgeschlagen${detail ? `: ${detail}` : ''}.`);
                return;
            }
            const data = await resp.json();
            const output = (data.output || '').trim();
            if (output) {
                result.textContent = output;
                result.classList.remove('empty');
                printBtn.disabled = false;
            } else {
                result.textContent = '(leere Übersetzung)';
                result.classList.add('empty');
                printBtn.disabled = true;
            }
        } catch {
            showError('Netzwerkfehler. Bitte erneut versuchen.');
        } finally {
            translateBtn.disabled = false;
            translateSpinner.style.display = 'none';
            translateLabel.textContent = 'Übersetzen';
        }
    });

    printBtn.addEventListener('click', () => window.print());

    // Auth + logout via the shared guard.
    LinguaGapAuth.requireUser();
    LinguaGapAuth.wireLogoutButton('logoutBtn');
})();
