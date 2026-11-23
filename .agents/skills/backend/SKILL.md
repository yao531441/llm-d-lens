---
name: backend
description: Use when changing Python APIs and domain services in apps/backend/src/llm_d_lens/, MCP, persistence, backend validation or task execution.
---

# Backend

Follow [AGENTS.md](../../../AGENTS.md) for discovery,
verification and unresolved decisions. All source paths below are repo-relative.

## Locate the implementation

- The Python service owns browser API routes, MCP, Playground and static frontend
  delivery. Inspect existing callers and compare error, timeout and streaming
  contracts before changing an integration.
- `apps/backend/src/llm_d_lens/`: Python APIs and domain services. Start in the owning domain
  (`configuration`, `deploy`, `evaluate`, `simulation`, etc.), then inspect
  `core/`, `schemas/`, `tasks/` and `utils/` for existing lower-level contracts.
- Use [the capability map](../../../.reuse/map.md) and current source to
  locate storage and Kubernetes helpers. A removed implementation or historical
  design document is not an available dependency.

## Kubernetes operations

This section owns the SDK/CLI operating contract. Prefer the existing `kubernetes_api` and
`kubernetes_mutations` helpers for structured operations; use
`utils.kubernetes.run_kubectl` / `scoped_runner` when callers need the existing
command-result contract. Inspect their error contracts before choosing:
`list_resources` retains legacy empty-list behavior on request failures.

Reuse cluster/session identity resolution, authentication, client pooling and
timeout/cancellation handling. Do not add scattered kubectl subprocess calls,
shell-string construction or separate SDK initialization for covered operations.
For session discovery, reuse the existing Python session helpers rather
than introducing another client or accepting arbitrary kubeconfig paths.

SDK request failures must not trigger CLI retries, especially for writes.
Unsupported command forms select CLI before execution. Keep domain validation
before transport; the retained CLI operations are listed below.

Tests must isolate SDK network access with local API fixtures or mocks at the
SDK boundary. A subprocess mock alone no longer isolates default Kubernetes
calls. Explicitly select CLI for CLI-specific tests; cover SDK behavior separately
and do not globally disable SDK merely to make a suite pass. Existing examples
are in `apps/backend/tests/unit/infrastructure/kubernetes/test_kubernetes_integration.py`.

### Backend selection and supported operations

The pinned official client is `kubernetes==36.0.3` (see
`apps/backend/pyproject.toml`). With neither backend variable set, recognized
reads and writes use its asynchronous SDK. `PRISM_KUBERNETES_BACKEND=sdk|cli`
overrides `PRISM_KUBERNETES_READ_BACKEND`; the latter enables SDK reads only.
Invalid values fail. Changing the environment requires a backend restart.
See the [environment reference](../../../docs/site/i18n/en/docusaurus-plugin-content-docs/current/environment-variables.mdx#kubernetes-and-networking)
for the selection table.

| Capability | SDK boundary |
| --- | --- |
| Resources | Core, apps, batch, storage, networking, RBAC and CRD definitions; custom resources use their served preferred API version |
| Queries | Named/list JSON or YAML; namespace/all namespaces; label/field selectors; pagination; multi-resource lists; selected raw API GETs |
| Writes | Namespace creation; one core/apps/batch manifest from stdin; strategic/merge/JSON patches; labels with overwrite checks; scale; cordon/uncordon; named/selector deletion and UID-aware deletion waits |
| Logs | Bounded named-Pod logs with container, tail, previous and timestamp selection |
| Guide planning | Session queries use the Python session helpers; default-context queries use the shared Kubernetes transport |
| Benchmark diagnostics | SDK Watch wakes failure checks; periodic pod/log reconciliation remains; CLI mode retains polling |

Only recognized argv shapes enter the adapter in
`apps/backend/src/llm_d_lens/infrastructure/kubernetes/commands.py`.
Unsupported flags, output formats and ambiguous selections choose CLI before
effects. Preserve raw Kubernetes JSON, `CommandResult` and caller-specific
list/error contracts. Typed-list items receive missing `kind`/`apiVersion` on
every page; preserve explicit item types and generic `List` responses.
Monitoring classification and Storage PVC/PV association depend on this.
Explicit JSON output remains JSON; default mutation output contains status lines,
not resource bodies or Secret data.

Keep Helm, Kustomize, **client-side kubectl apply**, unsupported rollout/wait,
Pod exec, port-forward, streaming/workload-selector logs, SSH, advanced
force/grace/dry-run flags and legacy `auth-provider` credentials on CLI paths.
Keep kubectl installed. Do not translate arbitrary scripts or replace client-side
apply with server-side apply: field ownership and pruning semantics differ.

### Credentials, connections and errors

- Reuse registry/session kubeconfig resolution. An unresolved explicit cluster
  must fail rather than inherit another default cluster; remote API reachability
  is not assumed.
- Reuse `infrastructure/kubernetes/proxy.py`: when a proxy is configured, add the
  selected API-server host to process `NO_PROXY`/`no_proxy` before SDK access.
  Preserve existing entries and skip explicit kubeconfig `proxy-url`. The
  aiohttp transport cannot rely on CIDR bypass entries as kubectl does.
- Support kubeconfig merging, context namespace, static token/token-file and TLS
  credentials through the existing auth helpers. ExecCredential plugins use argv
  arrays, noninteractive stdin, output/version/expiry validation, sanitized errors
  and bounded process-group cleanup on POSIX timeout/cancellation. Legacy
  `auth-provider` selects CLI before authentication. Never persist refreshed
  credentials back into kubeconfig.
- Reload configuration on each acquisition. Lifecycle-managed HTTP clients are
  pooled by credential/TLS identity, with 16-client and 16-borrower bounds,
  idle-only LRU eviction and lazy 30-second idle expiration. Shutdown closes
  clients and private exec-certificate files; outside that lifecycle, close per use.
- Watch bypasses the ordinary pool and reloads credentials on bounded reconnects.
  Bookmarks advance resourceVersion; 410 emits RESET and relists; 401/403 propagate;
  transient failures use bounded backoff. Early-exit consumers close their async
  iterator with `contextlib.aclosing`.
- Strict `query_resource` and mutation helpers preserve API exceptions.
  `list_resources` logs sanitized categories but retains legacy empty-list
  compatibility on request failures: an empty list does not prove absence.
  Scoped runners preserve timeout exceptions; deployment tuple results preserve
  exit-code/evidence behavior.

Isolated tests cover real SDK HTTP streams, mutual TLS, credential subprocesses,
pagination failures, credential/session rotation, cancellation/process descendants,
pool cleanup, writes and Watch reconnects. These fixtures do not establish live
write compatibility with every cluster, auth provider or CRD. Exercise SDK and
explicit CLI modes separately, including affected storage/evaluation/deployment
callers, and distinguish local coverage from real-cluster acceptance.

## Hardware profiles

Hardware-specific behavior is data-driven. Declare it in a profile
(`apps/backend/src/llm_d_lens/hardware/profiles/<vendor>.json`, shape in
`apps/backend/src/llm_d_lens/hardware/schema.json`) and, only when code is required, a provider
(`apps/backend/src/llm_d_lens/hardware/providers/<vendor>.py`) contributing driver
install/manifests and access `modes`, device resources, monitoring images and
dashboards, telemetry `device_metric_sources`, and the default runtime image.
Surface it through `GET /api/v1/hardware/capabilities` (see
`apps/backend/src/llm_d_lens/hardware/`) and the cluster overview `hardware` summary.

Do not hardcode a vendor list, a device-metric name/query, a DCGM/GFD/device-plugin
manifest, an access mode, a runtime image or an accelerator variant in a domain
service, and do not branch on a specific vendor (`if profile == 'intel-xpu'`).
Adding a hardware is adding a profile/provider, not editing
cluster/deploy/evaluate/simulation/monitoring/profiling code. Follow
[the hardware plugin architecture](../../../docs/design/hardware-plugin-architecture.md)
and [the AGENTS hardware rule](../../../AGENTS.md#hardware-is-profile-driven-never-hardcoded).
Cover the profile contract and its consumers, not only one vendor.

## Shared file placement and naming

| Type | Directory | Filename | Export |
| --- | --- | --- | --- |
| Assistant tool catalog (`/api/assistant/mcp`) | `apps/backend/src/llm_d_lens/assistant_tools/` | `snake_case.py` | snake_case function |
| Task MCP (`/api/mcp`) | `apps/backend/src/llm_d_lens/mcp/` | `snake_case.py` | snake_case function |
| Python shared base types | `apps/backend/src/llm_d_lens/core/` | `snake_case.py` | PascalCase class |
| Python HTTP adaptation | `apps/backend/src/llm_d_lens/api/` | `snake_case.py` | snake_case function |
| Python technical adapter | `apps/backend/src/llm_d_lens/infrastructure/<capability>/` | Responsibility-based `snake_case.py` | snake_case function; PascalCase class |
| Python domain capability | `apps/backend/src/llm_d_lens/<domain>/` | Responsibility-based `snake_case.py` | snake_case function; PascalCase class |
| Python domain models | Owning domain directory | `models.py` or existing model module | PascalCase class |
| Node test | Beside the tested module | `<filename>.test.ts` or existing `.test.js` convention | Existing test conventions |
| Python test | Owning domain or utility directory | `test_<capability>.py` | `test_<behavior>` function |

Examples: `apps/backend/src/llm_d_lens/core/exceptions.py` / `DomainError`,
`apps/backend/src/llm_d_lens/infrastructure/kubernetes/__init__.py` / `validate_namespace`.

Extend existing owners and preserve public import compatibility; do not rename
unrelated legacy modules. Persistence placement follows the DB PR's data-access
design; do not recreate the reverted JSON Store abstraction. Unresolved shared
schema/task ownership is not a reason to place new domain models in those folders.
The [capability map](../../../.reuse/map.md) links to current implementations,
callers and tests; do not maintain a second hand-written capability inventory.
Reuse existing upstream clients and preserve their timeout, parsing and
status/error mapping contracts. Python domain exceptions remain separate from HTTP serialization through
`install_problem_handlers`. Historical extraction reports do not establish
current persistence or shared-type ownership; consult the database skill and
the decision ledger rather than reviving a withdrawn file Store abstraction.

## File storage

Use the [storage scheme](../../../docs/design/storage-layout.md) for exact domain paths.
Reuse `storage_path` / `prism_temp_root` in `apps/backend/src/llm_d_lens/utils/paths.py`;
do not introduce another storage root
variable or restore legacy directory overrides. `LENS_*` roots take precedence
over XDG defaults. Explicit source checkout inputs such as `LLM_D_ROOT` retain
their existing input semantics.

- Keep persistent task files under `data/artifacts/<domain>/<owner-id>`, records
  under `data/metadata`, datasets under `data/datasets`, and credentials under
  `data/credentials`. Follow existing domain layouts for their exceptions.
- Use cache/log/scratch roots for their respective purposes. Model weights stay
  on the selected storage volume; local model-cache metadata is not the weights.
- Reuse `register_artifacts`, `artifact_uri` and `resolve_artifact` from
  `utils/artifact_store.py` for retained task files. Register source version,
  configuration IDs, checksum, truncation/completeness and retention class;
  mark unavailable provenance unknown. Exclude credentials from public manifests.
- Use owner IDs and relative paths for portable references, not host absolute
  paths as permanent identities. Preserve third-party output names and keep
  read/write/download/delete callers consistent with the same layout.
- Naming and manifest field details remain in the scheme. Adding a file type
  uses these existing primitives; it does not require a new storage subsystem.

## Preserve the boundaries

Keep request parsing and response translation at the API boundary; keep business
rules in the owning service. Compare status codes, error payloads, validation,
authentication context, cancellation and task lifecycle before changing a shared
contract. Inspect both Node and Python boundaries when requests cross services.
For persistence changes, inspect existing records, schema/migration requirements,
path ownership and affected readers before changing formats or locations.
Do not merge domain models solely because their fields look alike.

When an API change affects browser clients, also apply
[ui](../ui/SKILL.md). When it changes container, cluster,
startup or runtime configuration behavior, also apply
[deployment](../deployment/SKILL.md).

## Verify the affected contract

Use the domain's existing tests and `CONTRIBUTING.md` / `Makefile` commands
(`make test-js`, `make test-python`, `npm run type-check`, as applicable).
Cover affected callers and changed error paths. Report separately which checks
used mocks and which exercised configured external services or a real cluster.
