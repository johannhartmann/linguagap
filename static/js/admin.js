// Admin panel — CRUD for demo accounts.
//
// Talks to the /api/admin endpoints to list, create, edit, and delete the
// host login records that are persisted in /data/accounts.json. A 403 from
// any admin call sends the user back to /admin/login (session expired).

(() => {
    /** @type {Array<{email:string, password:string, display_name:string, logo_url:string}>} */
    let accounts = [];
    /** @type {string | null} */
    let editingEmail = null;

    /**
     * @param {string} msg
     */
    function showToast(msg) {
        const t = /** @type {HTMLElement} */ (document.getElementById('toast'));
        t.textContent = msg;
        t.classList.add('show');
        setTimeout(() => t.classList.remove('show'), 2500);
    }

    /**
     * Wrapper around fetch that handles auth-expiry redirect and JSON serialization.
     * @param {string} method
     * @param {string} url
     * @param {FormData | Record<string, unknown>} [body]
     * @returns {Promise<Response | null>}
     */
    async function api(method, url, body) {
        /** @type {RequestInit} */
        const opts = { method, headers: {} };
        if (body instanceof FormData) {
            opts.body = body;
        } else if (body) {
            /** @type {Record<string, string>} */ (opts.headers)['Content-Type'] =
                'application/json';
            opts.body = JSON.stringify(body);
        }
        const resp = await fetch(url, opts);
        if (resp.status === 403) {
            window.location.href = '/admin/login';
            return null;
        }
        return resp;
    }

    async function loadAccounts() {
        const resp = await api('GET', '/api/admin/accounts');
        if (!resp) return;
        accounts = await resp.json();
        renderTable();
    }

    /**
     * @param {string} tag
     * @param {Record<string, any> | null} [attrs]
     * @param {string | Element | Array<Element | null> | null} [children]
     * @returns {HTMLElement}
     */
    function el(tag, attrs, children) {
        const e = document.createElement(tag);
        if (attrs) {
            Object.entries(attrs).forEach(([k, v]) => {
                if (k === 'className') e.className = v;
                else if (k.startsWith('on')) e.addEventListener(k.slice(2).toLowerCase(), v);
                else e.setAttribute(k, v);
            });
        }
        if (children) {
            if (typeof children === 'string') e.textContent = children;
            else if (Array.isArray(children)) {
                children.forEach((c) => {
                    if (c) e.appendChild(c);
                });
            } else {
                e.appendChild(children);
            }
        }
        return e;
    }

    function renderTable() {
        const tbody = /** @type {HTMLElement} */ (document.getElementById('accountsBody'));
        tbody.replaceChildren();
        for (const a of accounts) {
            const tr = el('tr');
            const logoImg = el('img', {
                className: 'account-logo',
                src: a.logo_url,
                alt: '',
            });
            tr.appendChild(el('td', null, [logoImg]));

            if (editingEmail === a.email) {
                const emailInput = /** @type {HTMLInputElement} */ (
                    el('input', { className: 'edit-input', value: a.email })
                );
                const nameInput = /** @type {HTMLInputElement} */ (
                    el('input', { className: 'edit-input', value: a.display_name })
                );
                const pwInput = /** @type {HTMLInputElement} */ (
                    el('input', { className: 'edit-input', value: a.password })
                );
                tr.appendChild(el('td', null, [emailInput]));
                tr.appendChild(el('td', null, [nameInput]));
                tr.appendChild(el('td', null, [pwInput]));
                const saveBtn = el(
                    'button',
                    {
                        className: 'btn btn-primary btn-sm',
                        onClick: () => {
                            saveEdit(a.email, {
                                email: emailInput.value,
                                display_name: nameInput.value,
                                password: pwInput.value,
                                logo_url: a.logo_url,
                            });
                        },
                    },
                    'Save'
                );
                const cancelBtn = el(
                    'button',
                    { className: 'btn btn-secondary btn-sm', onClick: cancelEdit },
                    'Cancel'
                );
                tr.appendChild(el('td', { className: 'actions' }, [saveBtn, cancelBtn]));
            } else {
                tr.appendChild(el('td', null, a.email));
                tr.appendChild(el('td', null, a.display_name));
                tr.appendChild(el('td', null, '••••••••'));
                const editBtn = el(
                    'button',
                    {
                        className: 'btn btn-secondary btn-sm',
                        onClick: () => startEdit(a.email),
                    },
                    'Edit'
                );
                const delBtn = el(
                    'button',
                    {
                        className: 'btn btn-danger btn-sm',
                        onClick: () => deleteAccount(a.email),
                    },
                    'Delete'
                );
                tr.appendChild(el('td', { className: 'actions' }, [editBtn, delBtn]));
            }
            tbody.appendChild(tr);
        }
    }

    /** @param {string} email */
    function startEdit(email) {
        editingEmail = email;
        renderTable();
    }

    function cancelEdit() {
        editingEmail = null;
        renderTable();
    }

    /**
     * @param {string} originalEmail
     * @param {Record<string, string>} data
     */
    async function saveEdit(originalEmail, data) {
        const resp = await api(
            'PUT',
            `/api/admin/accounts/${encodeURIComponent(originalEmail)}`,
            data
        );
        if (resp?.ok) {
            editingEmail = null;
            showToast('Account updated');
            await loadAccounts();
        } else if (resp) {
            const err = await resp.json();
            showToast(err.detail || 'Error updating account');
        }
    }

    /** @param {string} email */
    async function deleteAccount(email) {
        if (!confirm(`Delete ${email}?`)) return;
        const resp = await api('DELETE', `/api/admin/accounts/${encodeURIComponent(email)}`);
        if (resp?.ok) {
            showToast('Account deleted');
            await loadAccounts();
        }
    }

    /**
     * @param {HTMLInputElement} fileInput
     * @returns {Promise<string | null>}
     */
    async function uploadLogo(fileInput) {
        if (!fileInput.files?.length) return null;
        const fd = new FormData();
        fd.append('file', fileInput.files[0]);
        const resp = await api('POST', '/api/admin/upload-logo', fd);
        if (resp?.ok) {
            return (await resp.json()).logo_url;
        }
        showToast('Logo upload failed');
        return null;
    }

    /** @type {HTMLFormElement} */
    const addForm = /** @type {HTMLFormElement} */ (document.getElementById('addForm'));
    addForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        let logoUrl = '/static/logos/synia.png';
        const logoInput = /** @type {HTMLInputElement} */ (document.getElementById('newLogo'));
        if (logoInput.files?.length) {
            const uploaded = await uploadLogo(logoInput);
            if (uploaded) logoUrl = uploaded;
        }
        const body = {
            email: /** @type {HTMLInputElement} */ (document.getElementById('newEmail')).value,
            password: /** @type {HTMLInputElement} */ (document.getElementById('newPassword'))
                .value,
            display_name: /** @type {HTMLInputElement} */ (
                document.getElementById('newDisplayName')
            ).value,
            logo_url: logoUrl,
        };
        const resp = await api('POST', '/api/admin/accounts', body);
        if (resp?.ok) {
            showToast('Account created');
            addForm.reset();
            /** @type {HTMLImageElement} */ (document.getElementById('newLogoPreview')).src =
                '/static/logos/synia.png';
            await loadAccounts();
        } else if (resp) {
            const err = await resp.json();
            showToast(err.detail || 'Error creating account');
        }
    });

    /** @type {HTMLInputElement} */
    const logoFileInput = /** @type {HTMLInputElement} */ (document.getElementById('newLogo'));
    logoFileInput.addEventListener('change', (e) => {
        const target = /** @type {HTMLInputElement} */ (e.target);
        const file = target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                /** @type {HTMLImageElement} */ (document.getElementById('newLogoPreview')).src =
                    /** @type {string} */ (ev.target?.result ?? '');
            };
            reader.readAsDataURL(file);
        }
    });

    /** @type {HTMLButtonElement} */
    const logoutBtn = /** @type {HTMLButtonElement} */ (document.getElementById('logoutBtn'));
    logoutBtn.addEventListener('click', async () => {
        await api('POST', '/api/admin/logout');
        window.location.href = '/admin/login';
    });

    loadAccounts();
})();
