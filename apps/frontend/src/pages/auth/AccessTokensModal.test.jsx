import assert from 'node:assert/strict';
import test from 'node:test';
import { createAccessToken, fetchAccessTokens, revokeAccessToken } from './authClient.js';

test('access token transport uses session credentials and CSRF, with no browser persistence', async () => {
    const previousFetch = globalThis.fetch;
    const previousDocument = globalThis.document;
    globalThis.document = { cookie: 'prism_csrf=test-csrf' };
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response(JSON.stringify(url.endsWith('/access-tokens') && options.method === 'POST'
            ? { id: 'one', token: 'lens-pat-once' } : []), { status: 200 });
    };
    try {
        assert.deepEqual(await fetchAccessTokens(), []);
        assert.equal((await createAccessToken('Agent', 7)).token, 'lens-pat-once');
        await revokeAccessToken('id/with/slash');
        assert.equal(calls[0].options.credentials, 'include');
        assert.equal(calls[1].options.headers.get('X-Prism-CSRF'), 'test-csrf');
        assert.deepEqual(JSON.parse(calls[1].options.body), { name: 'Agent', expiresInDays: 7 });
        assert.equal(calls[2].url, '/api/v1/auth/access-tokens/id%2Fwith%2Fslash');
        assert.equal(calls[2].options.method, 'DELETE');
        assert.equal(calls[2].options.headers.get('X-Prism-CSRF'), 'test-csrf');
    } finally {
        globalThis.fetch = previousFetch;
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});

test('access token API errors preserve server detail', async () => {
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'a login session is required' }), { status: 403 });
    try {
        await assert.rejects(createAccessToken('Agent', 30), /login session is required/);
    } finally {
        globalThis.fetch = previousFetch;
    }
});
