// Tiny i18n resolver shared by host.js and viewer.js.
//
// Both pages keep their own translation maps (the host's UI strings differ
// from the viewer's consent text), but the lookup logic — search the
// requested language, then English, then German, then the bare key — is
// identical. This module factors that logic out so neither page has to
// re-derive it.
//
// Usage:
//   const text = LinguaGapI18n.t([TRANSLATIONS], 'de', 'startRecording');
//   const text = LinguaGapI18n.t([I18N, PTT_I18N], foreignLang, 'pttHint');

(() => {
    /**
     * Resolve a translation key by walking the supplied maps in order, each
     * checked against (currentLang, 'en', 'de') before moving on. Unknown
     * keys fall back to the literal key string so the UI never blanks out.
     *
     * @param {Record<string, Record<string, string>>[]} maps
     * @param {string | null | undefined} currentLang
     * @param {string} key
     * @param {Record<string, string | number>} [replacements]
     * @returns {string}
     */
    function t(maps, currentLang, key, replacements = {}) {
        const langs = [currentLang, 'en', 'de'].filter(Boolean);
        for (const map of maps) {
            if (!map) continue;
            for (const lang of langs) {
                const value = map[lang]?.[key];
                if (typeof value === 'string') {
                    return applyReplacements(value, replacements);
                }
            }
        }
        return key;
    }

    /**
     * @param {string} text
     * @param {Record<string, string | number>} replacements
     */
    function applyReplacements(text, replacements) {
        for (const [k, v] of Object.entries(replacements)) {
            text = text.replace(`{${k}}`, String(v));
        }
        return text;
    }

    /** @type {any} */ (window).LinguaGapI18n = { t };
})();
