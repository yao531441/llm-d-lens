import assert from 'node:assert/strict';
import test from 'node:test';
import { listDeploymentVerifications, startDeploymentVerification } from '../../api/agenticDeploymentBackend.js';

test('verification transport uses canonical execution ids, credentials and CSRF', async () => {
    const previousFetch = globalThis.fetch;
    const previousDocument = globalThis.document;
    const calls = [];
    globalThis.document = { cookie: 'prism_csrf=verification-csrf' };
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response(JSON.stringify(options.method === 'POST' ? { id: 'check-1', status: 'queued' }
            : { items: [{ id: 'check-1', verdict: 'inconclusive' }] }), { status: options.method === 'POST' ? 202 : 200 });
    };
    try {
        assert.equal((await startDeploymentVerification('exec/1', ['chat'], 3)).id, 'check-1');
        assert.equal((await listDeploymentVerifications('exec/1')).items[0].verdict, 'inconclusive');
        assert.equal(calls[0].url, '/api/v1/deployments/executions/exec%2F1/verifications');
        assert.equal(calls[0].options.credentials, 'include');
        assert.equal(calls[0].options.headers.get('X-Prism-CSRF'), 'verification-csrf');
        assert.deepEqual(JSON.parse(calls[0].options.body), { checks: ['chat'], requests: 3 });
    } finally {
        globalThis.fetch = previousFetch;
        if (previousDocument === undefined) delete globalThis.document;
        else globalThis.document = previousDocument;
    }
});

test('verification permission and conflict errors preserve server detail', async () => {
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'verification already active' }), { status: 409 });
    try {
        await assert.rejects(startDeploymentVerification('exec-1', ['chat'], 1), /verification already active/);
    } finally {
        globalThis.fetch = previousFetch;
    }
});
