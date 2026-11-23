import { useCallback, useEffect, useRef, useState } from 'react';
import { usePolling } from '../../../../shared/lib/usePolling';
import { useSubmission } from '../../../../shared/lib/useSubmission';
import { Panel } from '../../../../shared/ui/Panel';
import { Button } from '../../../../shared/ui/Button';
import { Badge } from '../../../../shared/ui/Badge';
import { Checkbox, Input, Label } from '../../../../shared/ui/FormControls';
import { FormError } from '../../../../shared/ui/FormError';
import { listDeploymentVerifications, startDeploymentVerification } from '../../api/agenticDeploymentBackend';

const CHECKS = ['chat', 'prefix_cache'];

export function DeploymentVerificationPanel({ executionId, status, canConnect = false }) {
    const [records, setRecords] = useState([]);
    const [error, setError] = useState('');
    const [checks, setChecks] = useState(['chat']);
    const [requests, setRequests] = useState('10');
    const controller = useRef(null);
    const { pending, error: submitError, run } = useSubmission('Failed to start verification');
    const load = useCallback(async () => {
        controller.current?.abort();
        const request = new AbortController();
        controller.current = request;
        try {
            const data = await listDeploymentVerifications(executionId, request.signal);
            if (!request.signal.aborted) { setRecords(data.items || []); setError(''); }
        } catch (failure) {
            if (!request.signal.aborted) setError(failure.message);
        }
    }, [executionId]);
    useEffect(() => () => controller.current?.abort(), []);
    usePolling(load, { immediate: true });
    const budget = Number(requests);
    const invalid = !checks.length || !Number.isInteger(budget) || budget < 1 || budget > 20
        || (checks.includes('prefix_cache') && budget < 10);
    const active = records.some((item) => ['queued', 'running'].includes(item.status));
    const start = () => run(async () => {
        await startDeploymentVerification(executionId, checks, budget);
        await load();
    });
    return (
        <Panel title="Deployment verification">
            <div className="space-y-3 text-theme-text">
                <p className="text-xs text-theme-muted">Small inference checks run on the Lens host and share persisted results with MCP.
                    Requests may warm caches. Transfer, rank and routing verification are not supported.</p>
                {canConnect && (
                    <div className="space-y-2">
                        <div className="flex flex-wrap gap-3">
                            {CHECKS.map((check) => (
                                <Label key={check}>
                                    <Checkbox checked={checks.includes(check)} disabled={pending || active}
                                        onChange={(event) => setChecks((current) => event.target.checked
                                            ? [...current, check] : current.filter((value) => value !== check))} /> {check}
                                </Label>
                            ))}
                        </div>
                        <Label htmlFor={`verification-budget-${executionId}`}>Total request budget (1–20; prefix cache needs at least 10)</Label>
                        <Input id={`verification-budget-${executionId}`} type="number" min="1" max="20" value={requests}
                            disabled={pending || active} onChange={(event) => setRequests(event.target.value)} />
                        <Button size="sm" onClick={start} isLoading={pending}
                            disabled={invalid || active || status !== 'ready'}>Run verification</Button>
                    </div>
                )}
                <FormError message={error || submitError} />
                {!records.length && !error && <p className="text-xs text-theme-muted">No saved verifications.</p>}
                {records.map((record) => (
                    <div key={record.id} className="space-y-1 rounded border border-theme-border p-3">
                        <div className="flex flex-wrap gap-2">
                            <Badge tone={record.status === 'failed' ? 'danger' : 'neutral'}>{record.status}</Badge>
                            {record.verdict && <Badge tone={record.verdict === 'pass' ? 'success' : record.verdict === 'fail' ? 'danger' : 'warning'}>{record.verdict}</Badge>}
                            <span className="text-xs text-theme-muted">{record.created_at}</span>
                        </div>
                        <p className="break-all font-mono text-xs">{record.id}</p>
                        <p className="text-xs">Sent {record.requests_sent}/{record.request.requests}; completed {record.requests_completed};
                            successful {record.requests_succeeded}; reported output tokens {record.output_tokens} (budget {record.max_output_tokens}).</p>
                        {record.error && <p className="text-xs text-theme-muted">{record.error}</p>}
                        {(record.checks || []).map((check) => (
                            <div key={check.check} className="text-xs">
                                <strong>{check.check}: {check.verdict}</strong> — {check.evidence.join(' ')}
                            </div>
                        ))}
                    </div>
                ))}
                <p className="text-xs text-theme-muted">The latest 20 records are retained with this execution. Task success means checks completed;
                    the verdict separately reports what the evidence proves.</p>
            </div>
        </Panel>
    );
}
