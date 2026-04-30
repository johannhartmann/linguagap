// User login page — POSTs credentials to /api/login and redirects on success.

(() => {
    const form = /** @type {HTMLFormElement | null} */ (document.getElementById('loginForm'));
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = /** @type {HTMLButtonElement} */ (document.getElementById('loginBtn'));
        const errorMsg = /** @type {HTMLElement} */ (document.getElementById('errorMsg'));
        const emailInput = /** @type {HTMLInputElement} */ (document.getElementById('email'));
        const passwordInput = /** @type {HTMLInputElement} */ (document.getElementById('password'));

        errorMsg.textContent = '';
        btn.disabled = true;
        btn.textContent = 'Signing in…';

        try {
            const resp = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email: emailInput.value,
                    password: passwordInput.value,
                }),
            });
            if (resp.ok) {
                window.location.href = '/';
            } else {
                const data = await resp.json();
                errorMsg.textContent = data.error || 'Invalid username or password';
            }
        } catch {
            errorMsg.textContent = 'Connection error. Please try again.';
        } finally {
            btn.disabled = false;
            btn.textContent = 'Sign In';
        }
    });
})();
