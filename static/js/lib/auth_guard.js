// Auth guard — shared between every authenticated page.
//
// Each page used to roll its own copy of the same two patterns:
//   1. Hit /api/me on load and redirect to /login if the session is gone.
//   2. Wire a logout button to POST /api/logout and bounce back to /login.
//
// This module exposes them on window.LinguaGapAuth so host.js, viewer.js,
// translate.js, and any future pages can share a single implementation.

(() => {
    /** @typedef {{email:string, display_name:string, logo_url:string, is_admin:boolean}} CurrentUser */

    /**
     * Verify the user is signed in. On 401 redirects to /login (and resolves
     * with null so the caller can short-circuit). On any other error returns
     * null without redirecting — the caller can decide what to do.
     *
     * @returns {Promise<CurrentUser | null>}
     */
    async function requireUser() {
        try {
            const resp = await fetch('/api/me');
            if (!resp.ok) {
                window.location.href = '/login';
                return null;
            }
            return await resp.json();
        } catch (e) {
            console.error('Auth check failed:', e);
            return null;
        }
    }

    /**
     * Attach a click handler that POSTs to /api/logout and redirects.
     *
     * @param {string} buttonId  HTML id of the logout button.
     * @param {string} [redirectTo]  Defaults to /login.
     */
    function wireLogoutButton(buttonId, redirectTo = '/login') {
        const btn = /** @type {HTMLButtonElement | null} */ (document.getElementById(buttonId));
        if (!btn) return;
        btn.addEventListener('click', async () => {
            await fetch('/api/logout', { method: 'POST' });
            window.location.href = redirectTo;
        });
    }

    /** @type {any} */ (window).LinguaGapAuth = { requireUser, wireLogoutButton };
})();
