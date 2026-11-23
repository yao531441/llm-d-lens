import { useEffect, useRef, useState } from 'react';
import { Modal } from '../../shared/ui/Modal';
import { Button } from '../../shared/ui/Button';
import { Input, Label } from '../../shared/ui/FormControls';
import { FormError } from '../../shared/ui/FormError';
import { EmptyState } from '../../shared/ui/EmptyState';
import { LoadingState } from '../../shared/ui/Spinner';
import { useClipboard } from '../../shared/lib/useClipboard';
import { usePolling } from '../../shared/lib/usePolling';
import { useSubmission } from '../../shared/lib/useSubmission';
import { createAccessToken, fetchAccessTokens, revokeAccessToken } from './authClient';

export function AccessTokensModal({ onClose }) {
    const [items, setItems] = useState([]);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState('');
    const [name, setName] = useState('');
    const [days, setDays] = useState(30);
    const [secret, setSecret] = useState('');
    const alive = useRef(true);
    const { pending, error, setError, run } = useSubmission('Failed to manage access tokens');
    const { copied, copy } = useClipboard();
    const load = async () => {
        try {
            const records = await fetchAccessTokens();
            if (alive.current) { setItems(records); setLoadError(''); }
        } catch (failure) {
            if (alive.current) setLoadError(failure.message || 'Failed to load access tokens');
        } finally {
            if (alive.current) setLoading(false);
        }
    };
    useEffect(() => {
        alive.current = true;
        load();
        return () => { alive.current = false; };
    }, []);
    usePolling(load, { intervalMs: 10000 });
    const create = async (event) => {
        event.preventDefault();
        await run(async () => {
            const record = await createAccessToken(name.trim(), Number(days));
            setSecret(record.token);
            setName('');
            await load();
        });
    };
    const revoke = (id) => run(async () => {
        await revokeAccessToken(id);
        await load();
    });
    return (
        <Modal isOpen title="Personal access tokens" subtitle="Connect MCP clients using your current Lens permissions."
            onClose={pending ? undefined : onClose} closeOnBackdrop={!pending} closeOnEscape={!pending}
            footer={<Button variant="secondary" onClick={onClose} disabled={pending}>Close</Button>}>
            <div className="flex flex-col gap-4 text-theme-text">
                <p className="text-sm text-theme-muted">Tokens expire on a fixed date and can be revoked individually. Password changes and account disabling revoke them. Tokens cannot manage credentials.</p>
                {secret && <section aria-label="New access token" className="rounded border border-theme-border p-3">
                    <p className="text-sm font-semibold">Copy this token now. It will only be shown once.</p>
                    <Input aria-label="New token" readOnly value={secret} className="mt-2 font-mono" />
                    <div className="mt-2 flex gap-2">
                        <Button size="sm" onClick={async () => { if (!await copy(secret)) setError('Could not copy token. Select and copy it manually.'); }}>{copied ? 'Copied' : 'Copy token'}</Button>
                        <Button size="sm" variant="secondary" onClick={() => setSecret('')}>Hide token</Button>
                    </div>
                </section>}
                <form onSubmit={create} className="flex flex-col gap-3">
                    <div><Label htmlFor="access-token-name">Name</Label><Input id="access-token-name" value={name} onChange={(event) => setName(event.target.value)} required maxLength={100} /></div>
                    <div><Label htmlFor="access-token-days">Expires in (days)</Label><Input id="access-token-days" type="number" min={1} max={365} value={days} onChange={(event) => setDays(event.target.value)} required /></div>
                    <Button type="submit" variant="primary" isLoading={pending} disabled={!name.trim() || Boolean(secret)}>Create token</Button>
                </form>
                <FormError message={error || loadError} />
                {loadError && <Button variant="secondary" onClick={load}>Retry loading</Button>}
                {loading ? <LoadingState label="Loading access tokens" /> : items.length === 0 ? <EmptyState title="No access tokens" message="Create a token to connect an MCP client." /> :
                    <ul className="flex flex-col gap-2">{items.map((item) => {
                        const status = item.revoked_at ? 'Revoked' : new Date(item.expires_at) <= new Date() ? 'Expired' : 'Active';
                        return <li key={item.id} className="rounded border border-theme-border p-3 text-sm">
                            <div className="flex items-center justify-between gap-2"><strong>{item.name}</strong><span>{status}</span></div>
                            <p className="text-theme-muted">{item.token_prefix}…</p>
                            <p className="text-theme-muted">Expires: {new Date(item.expires_at).toLocaleString()}</p>
                            <p className="text-theme-muted">Last used: {item.last_used_at ? new Date(item.last_used_at).toLocaleString() : 'Never'}</p>
                            {status === 'Active' && <Button size="sm" variant="dangerOutline" disabled={pending} onClick={() => revoke(item.id)}>Revoke {item.name}</Button>}
                        </li>;
                    })}</ul>}
            </div>
        </Modal>
    );
}
