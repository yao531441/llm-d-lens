import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, copyFileSync, symlinkSync, existsSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';

test('installer legacy backend port selects the unified port and resolves collisions once', t => {
  const { root } = setup(t);
  const result = spawnSync('bash', ['-c', `
    source "$1"
    parse_args --app-port 3000 --backend-port 4100
    port_in_use() { [[ "$1" == 4100 ]]; }
    describe_port_owner() { echo test-process; }
    find_free_port() { echo 4101; }
    resolve_ports_silently
    printf '%s:%s\\n' "$APP_PORT" "$BACKEND_PORT"
  `, 'test', resolve('scripts/LensInstaller-Ubuntu-x86_64-online.sh')], {
    cwd: root, encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), '4101:4101');
  assert.equal((result.stderr.match(/switched/g) || []).length, 1);
});

test('installed controller refuses startup when the Python MCP catalog fails', t => {
  const { root, put } = setup(t);
  execFileSync('bash', ['-c', 'source "$1"; INSTALL_DIR="$2"; write_control_script',
    'test', resolve('scripts/LensInstaller-Ubuntu-x86_64-online.sh'), root]);
  execFileSync('bash', ['-n', join(root, 'lensctl.sh')]);
  put('.lens-install.env', 'APP_PORT=49191\nBACKEND_PORT=8081\nENABLE_HTTPS=0\n');
  put('.venv/bin/python', `#!/usr/bin/env bash
if [[ "$1" == -m && "$2" == llm_d_lens.infrastructure.hostos ]]; then
  case "$3" in
    check) exit 0 ;;
    port-in-use) exit 1 ;;
    free-port) printf '%s\\n' "$5"; exit 0 ;;
    lan-ip) exit 0 ;;
    *) exit 0 ;;
  esac
fi
exit 1
`);
  chmodSync(join(root, '.venv/bin/python'), 0o755);
  const result = spawnSync('bash', [join(root, 'lensctl.sh'), 'start'], {
    encoding: 'utf8',
    env: { ...process.env, LENS_DATA_DIR: join(root, 'data'), LENS_CACHE_DIR: join(root, 'cache'),
      LENS_LOG_DIR: join(root, 'logs'), LENS_SCRATCH_DIR: join(root, 'scratch') },
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /MCP tool catalog validation failed/);
  assert.equal(existsSync(join(root, '.prism-pids/app.pid')), false);
  assert.equal(existsSync(join(root, '.prism-pids/backend.pid')), false);
  const config = readFileSync(join(root, '.lens-install.env'), 'utf8');
  assert.equal(config.match(/^APP_PORT=(\d+)$/m)[1], config.match(/^BACKEND_PORT=(\d+)$/m)[1]);
});

test('unified HTTPS launcher defaults the MCP trust certificate and preserves explicit overrides', t => {
  const { root, put } = setup(t);
  copyFileSync('hack/run-backend.sh', join(root, 'hack/run-backend.sh'));
  put('bin/python', `#!/usr/bin/env bash
if [[ "$1" == -c ]]; then exit 0; fi
if [[ "$2" == llm_d_lens.db.bootstrap_cli ]]; then exit "\${LENS_TEST_BOOTSTRAP_EXIT:-0}"; fi
printf '%s\\n' "$PRISM_MCP_URL" "$PRISM_MCP_CA_FILE" "$@"
`);
  chmodSync(join(root, 'bin/python'), 0o755);
  for (const customCa of ['', '/custom/ca.pem']) {
    const result = spawnSync('bash', [join(root, 'hack/run-backend.sh')], {
      encoding: 'utf8',
      env: { ...process.env, PYTHON_BIN: join(root, 'bin/python'), BACKEND_PORT: '4100',
        TLS_CERT_FILE: '/cert with spaces.pem', TLS_KEY_FILE: '/key.pem', TLS_CA_FILE: '',
        PRISM_MCP_URL: '', PRISM_MCP_CA_FILE: customCa,
        LENS_DATA_DIR: join(root, 'data'), LENS_CACHE_DIR: join(root, 'cache'),
        LENS_LOG_DIR: join(root, 'logs'), LENS_SCRATCH_DIR: join(root, 'scratch') },
    });
    assert.equal(result.status, 0, result.stderr);
    const lines = result.stdout.trim().split('\n');
    assert.deepEqual(lines.slice(0, 2), ['https://127.0.0.1:4100/api/assistant/mcp', customCa || '/cert with spaces.pem']);
    assert.equal(lines[lines.indexOf('--ssl-certfile') + 1], '/cert with spaces.pem');
    assert.equal(lines[lines.indexOf('--port') + 1], '4100');
  }
  const failed = spawnSync('bash', [join(root, 'hack/run-backend.sh')], {
    encoding: 'utf8',
    env: { ...process.env, PYTHON_BIN: join(root, 'bin/python'), LENS_TEST_BOOTSTRAP_EXIT: '42',
      LENS_DATA_DIR: join(root, 'data'), LENS_CACHE_DIR: join(root, 'cache'),
      LENS_LOG_DIR: join(root, 'logs'), LENS_SCRATCH_DIR: join(root, 'scratch') },
  });
  assert.equal(failed.status, 42);
  assert.equal(failed.stdout, '');
});

test('installed controller mirrors HTTPS into the Gateway authz TLS override', t => {
  const cases = [
    ['HTTPS with an existing certificate', { https: 1, cert: true }, {}, '1'],
    ['plain HTTP', { https: 0, cert: false }, {}, '0'],
    ['explicit env wins over HTTPS', { https: 1, cert: true }, { LENS_MODEL_GATEWAY_AUTHZ_TLS_INSECURE: '0' }, '0'],
    ['explicit env wins over plain HTTP', { https: 0, cert: false }, { LENS_MODEL_GATEWAY_AUTHZ_TLS_INSECURE: '1' }, '1'],
    ['HTTPS downgraded when no certificate host is detectable', { https: 1, cert: false }, {}, '0'],
  ];
  for (const [name, install, env, expected] of cases) {
    const { root, put } = setup(t);
    execFileSync('bash', ['-c', 'source "$1"; INSTALL_DIR="$2"; write_control_script',
      'test', resolve('scripts/LensInstaller-Ubuntu-x86_64-online.sh'), root]);
    execFileSync('bash', ['-n', join(root, 'lensctl.sh')]);
    const marker = join(root, 'gateway-authz-tls-insecure');
    let tls = '';
    if (install.cert) {
      put('certs/tls.crt', 'certificate');
      put('certs/tls.key', 'key');
      tls = `TLS_CERT_FILE=${join(root, 'certs/tls.crt')}\nTLS_KEY_FILE=${join(root, 'certs/tls.key')}\n`;
    }
    put('.lens-install.env', `APP_PORT=49191\nBACKEND_PORT=49191\nENABLE_HTTPS=${install.https}\n${tls}`);
    put('.venv/bin/python', '#!/usr/bin/env bash\nexit 0\n');
    chmodSync(join(root, '.venv/bin/python'), 0o755);
    put('hack/run-backend.sh', '#!/usr/bin/env bash\nprintf \'%s\\n\' "${LENS_MODEL_GATEWAY_AUTHZ_TLS_INSECURE:-<unset>}" > "$LENS_TEST_MARKER"\nexit 0\n');
    for (const [tool, body] of [
      ['ss', '#!/usr/bin/env bash\nexit 0\n'],
      ['hostname', '#!/usr/bin/env bash\nexit 0\n'],
      ['sleep', '#!/usr/bin/env bash\nexit 0\n'],
      ['curl', '#!/usr/bin/env bash\nfor i in $(seq 1 100); do [[ -s "$LENS_TEST_MARKER" ]] && exit 0; /bin/sleep 0.05; done\nexit 1\n'],
    ]) {
      put(`bin/${tool}`, body);
      chmodSync(join(root, `bin/${tool}`), 0o755);
    }
    const result = spawnSync('bash', [join(root, 'lensctl.sh'), 'start'], {
      encoding: 'utf8',
      env: { ...process.env, PATH: `${join(root, 'bin')}:${process.env.PATH}`,
        LENS_TEST_MARKER: marker, LENS_DATA_DIR: join(root, 'data'),
        LENS_CACHE_DIR: join(root, 'cache'), LENS_LOG_DIR: join(root, 'logs'),
        LENS_SCRATCH_DIR: join(root, 'scratch'),
        LENS_MODEL_GATEWAY_AUTHZ_TLS_INSECURE: undefined,
        TLS_CERT_FILE: '', TLS_KEY_FILE: '', APP_HOST: undefined, LENS_IN_PLACE: undefined,
        ...env },
    });
    assert.equal(result.status, 0, `${name}: ${result.stdout}${result.stderr}`);
    assert.equal(readFileSync(marker, 'utf8').trim(), expected, name);
  }
});

function setup(t) {
  const root = mkdtempSync(join(tmpdir(), 'reuse-dev-start-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const put = (path, text) => { mkdirSync(resolve(root, path, '..'), { recursive: true }); writeFileSync(join(root, path), text); };
  put('src/a.js', 'export function load() { return 1; }');
  put('.reuse/catalog.json', JSON.stringify({ version: 1, entries: [{ id: 'load', path: 'src/a.js', symbols: ['load'], purpose: 'Load', boundaries: 'Local only', examples: ['src/a.js'], tests: [], testNote: 'None' }] }));
  put('.reuse/map.md', 'stale');
  mkdirSync(join(root, 'hack'));
  copyFileSync('hack/lens-local.sh', join(root, 'hack/lens-local.sh'));
  copyFileSync('hack/storage-env.sh', join(root, 'hack/storage-env.sh'));
  symlinkSync(resolve('hack/reuse'), join(root, 'hack/reuse'));
  mkdirSync(join(root, 'llm-d/guides'), { recursive: true });
  put('bin/kubectl', `#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$LENS_TEST_ROOT/kubectl-calls"
if [[ "\${LENS_TEST_KUBECTL_EXIT:-0}" != 0 ]]; then
  echo "Cluster unavailable" >&2
  exit "$LENS_TEST_KUBECTL_EXIT"
fi
printf '%s\\n' "$LENS_TEST_NODE_JSON"
`);
  chmodSync(join(root, 'bin/kubectl'), 0o755);
  execFileSync('git', ['init', '-q', root]);
  execFileSync('git', ['add', '.'], { cwd: root });
  execFileSync('git', ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'baseline'], { cwd: root });
  // Only service/environment boundaries are replaced. Real shell dispatch,
  // map generation, Git scope, validation and decision checks execute.
  const run = (command, { args = [], env = {}, overrides = '', cwd = join(root, 'hack') } = {}) => spawnSync('bash', ['-c', `
    source "$1"
    shift
    ensure_node() { :; }
    ensure_node_modules() { :; }
    ensure_venv() { :; }
    cmd_check() { echo check >> "$ROOT_DIR/events"; }
    cmd_stop() { echo stop >> "$ROOT_DIR/events"; }
    cmd_clean() { printf 'clean:%s\\n' "$1" >> "$ROOT_DIR/events"; }
    start_services() {
      echo start >> "$ROOT_DIR/events"
      "$LENS_TEST_NODE" -e 'console.log(JSON.stringify(Object.fromEntries(["TARGET_NODE","PRISM_K8S_TARGET_NODE","KUBECONFIG","PRISM_GPU_PCI_ALLOWLIST","LLM_D_BENCH_NAMESPACE_PREFIX","LLM_D_BENCH_KUBECTL_TIMEOUT_SECONDS"].map(key => [key, process.env[key] ?? null]))))' > "$ROOT_DIR/exported.json"
    }
    ${overrides}
    lens_local_main "$@"
  `, 'test', join(root, 'hack/lens-local.sh'), command, ...args], {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      TARGET_NODE: '',
      PRISM_K8S_TARGET_NODE: '',
      PRISM_GPU_PCI_ALLOWLIST: '',
      KUBECONFIG: undefined,
      LLM_D_BENCH_NAMESPACE_PREFIX: undefined,
      LLM_D_BENCH_KUBECTL_TIMEOUT_SECONDS: undefined,
      LLM_D_BENCH_KUBECTL_PATH: join(root, 'bin/kubectl'),
      LLM_D_ROOT: join(root, 'llm-d'),
      LENS_TEST_ROOT: root,
      LENS_TEST_NODE: process.execPath,
      LENS_TEST_NODE_JSON: JSON.stringify({ metadata: { name: 'node-a' }, spec: {}, status: { conditions: [{ type: 'Ready', status: 'True' }] } }),
      ...env,
    },
  });
  return { root, put, run };
}

test('restart from hack regenerates the map and validates it before restarting services', t => {
  const { root, run } = setup(t);
  const result = run('restart');
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(readFileSync(join(root, '.reuse/map.md'), 'utf8'), /src\/a.js/);
  assert.equal(existsSync(join(root, 'docs')), false);
  assert.equal(readFileSync(join(root, 'events'), 'utf8'), 'stop\nstart\n');
  assert.equal(existsSync(join(root, '.cache/reuse/startup-report.json')), false);
});

test('target flags override environment and export constraints without injecting deployment defaults', t => {
  const { root, run } = setup(t);
  const result = run('restart', {
    args: ['--target-node', 'node-a', '--kubeconfig', 'cluster config.yaml'],
    env: { TARGET_NODE: 'node-b', PRISM_K8S_TARGET_NODE: 'node-c', KUBECONFIG: '/old/config', PRISM_GPU_PCI_ALLOWLIST: '0000:01:00.0' },
  });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.equal(readFileSync(join(root, 'events'), 'utf8'), 'stop\nstart\n');
  assert.equal(readFileSync(join(root, 'kubectl-calls'), 'utf8'), '--request-timeout=30s get node node-a -o json\n');
  assert.deepEqual(JSON.parse(readFileSync(join(root, 'exported.json'))), {
    TARGET_NODE: 'node-a',
    PRISM_K8S_TARGET_NODE: 'node-a',
    KUBECONFIG: join(root, 'hack/cluster config.yaml'),
    PRISM_GPU_PCI_ALLOWLIST: '0000:01:00.0',
    LLM_D_BENCH_NAMESPACE_PREFIX: null,
    LLM_D_BENCH_KUBECTL_TIMEOUT_SECONDS: null,
  });
});

test('environment target mode preserves kubeconfig file lists and leaves device selection to the backend', t => {
  const { root, run } = setup(t);
  const result = run('start', { env: { TARGET_NODE: 'node-a', KUBECONFIG: '/cluster/a:/cluster/b' }, cwd: root });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  const exported = JSON.parse(readFileSync(join(root, 'exported.json')));
  assert.equal(exported.PRISM_K8S_TARGET_NODE, 'node-a');
  assert.equal(exported.KUBECONFIG, '/cluster/a:/cluster/b');
  assert.equal(exported.PRISM_GPU_PCI_ALLOWLIST, '');
  assert.equal(readFileSync(join(root, 'kubectl-calls'), 'utf8').split('\n').filter(Boolean).length, 1);
});

test('legacy target environment remains supported and check never starts services', t => {
  const { root, run } = setup(t);
  const result = run('check', { env: { PRISM_K8S_TARGET_NODE: 'node-a' } });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.equal(readFileSync(join(root, 'events'), 'utf8'), 'check\n');
  assert.ok(existsSync(join(root, 'kubectl-calls')));
});

for (const command of ['start', 'check']) test(`${command} without a target does not query Kubernetes or require a Guide checkout`, t => {
  const { root, run } = setup(t);
  const result = run(command, { env: { LLM_D_ROOT: '/missing', LLM_D_BENCH_KUBECTL_PATH: '/missing' } });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.ok(!existsSync(join(root, 'kubectl-calls')));
  if (command === 'start') assert.equal(JSON.parse(readFileSync(join(root, 'exported.json'))).KUBECONFIG, null);
});

for (const [name, env, overrides, message] of [
  ['unreachable cluster', { LENS_TEST_KUBECTL_EXIT: '1' }, '', /not accessible/],
  ['not ready', { LENS_TEST_NODE_JSON: '{"metadata":{"name":"node-a"},"status":{"conditions":[{"type":"Ready","status":"False"}]}}' }, '', /Ready and not cordoned/],
  ['cordoned', { LENS_TEST_NODE_JSON: '{"metadata":{"name":"node-a"},"spec":{"unschedulable":true},"status":{"conditions":[{"type":"Ready","status":"True"}]}}' }, '', /Ready and not cordoned/],
  ['missing conditions', { LENS_TEST_NODE_JSON: '{"metadata":{"name":"node-a"}}' }, '', /Ready and not cordoned/],
  ['wrong node', { LENS_TEST_NODE_JSON: '{"metadata":{"name":"node-b"},"status":{"conditions":[{"type":"Ready","status":"True"}]}}' }, '', /Ready and not cordoned/],
  ['malformed response', { LENS_TEST_NODE_JSON: 'not-json' }, '', /Ready and not cordoned/],
  ['missing kubectl', { LLM_D_BENCH_KUBECTL_PATH: '/missing' }, '', /executable kubectl/],
  ['missing jq', {}, 'command() { if [[ "$*" == "-v jq" ]]; then return 1; fi; builtin command "$@"; }', /require jq/],
  ['missing checkout', { LLM_D_ROOT: '/missing' }, '', /containing guides/],
]) test(`restart rejects ${name} before stopping existing services`, t => {
  const { root, run } = setup(t);
  const result = run('restart', { env: { TARGET_NODE: 'node-a', ...env }, overrides });
  assert.equal(result.status, 1, result.stdout + result.stderr);
  assert.match(result.stdout + result.stderr, message);
  assert.ok(!existsSync(join(root, 'events')));
});

for (const [command, args, status] of [
  ['restart', ['--target-node'], 2],
  ['restart', ['--target-node', ''], 2],
  ['restart', ['--kubeconfig', '--target-node', 'node-a'], 2],
  ['restart', ['--target-node', 'INVALID_NODE'], 1],
  ['restart', ['--target-config', 'unused.env'], 2],
  ['restart', ['--yes'], 2],
  ['stop', ['--target-node', 'node-a'], 2],
  ['unknown', [], 2],
]) test(`invalid invocation is rejected: ${command} ${args.join(' ')}`, t => {
  const { root, run } = setup(t);
  const result = run(command, { args });
  assert.equal(result.status, status, result.stdout + result.stderr);
  assert.ok(!existsSync(join(root, 'events')));
  assert.ok(!existsSync(join(root, 'kubectl-calls')));
});

for (const command of ['stop', 'status', 'clean', 'install-node', '--help']) test(`${command} remains independent of target-node connectivity`, t => {
  const { root, run } = setup(t);
  const result = run(command, { args: command === 'clean' ? ['--yes'] : [], env: { TARGET_NODE: 'INVALID_NODE', LLM_D_BENCH_KUBECTL_PATH: '/missing' } });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.ok(!existsSync(join(root, 'kubectl-calls')));
  if (command === 'clean') assert.equal(readFileSync(join(root, 'events'), 'utf8'), 'clean:--yes\n');
});

for (const [args, expected] of [
  [['--yes'], 'clean:--yes:default'],
  [['--cache', '--yes'], 'clean:--yes:cache'],
  [['--hard'], 'clean::hard'],
  [['--all', '--yes'], 'clean:--yes:all'],
]) test(`clean forwards ${args.join(' ')} without target-node checks`, t => {
  const { root, run } = setup(t);
  const result = run('clean', {
    args,
    env: { TARGET_NODE: 'INVALID_NODE', LLM_D_BENCH_KUBECTL_PATH: '/missing' },
    overrides: 'cmd_clean() { printf "clean:%s:%s\\n" "$1" "$2" >> "$ROOT_DIR/events"; }',
  });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.ok(!existsSync(join(root, 'kubectl-calls')));
  assert.equal(readFileSync(join(root, 'events'), 'utf8'), `${expected}\n`);
});

test('invalid registration stops restart before touching running services', t => {
  const { root, put, run } = setup(t);
  put('src/a.js', 'export const other = 1;');
  const result = run('restart');
  assert.equal(result.status, 1);
  assert.ok(!existsSync(join(root, 'events')));
  assert.match(result.stdout + result.stderr, /load/);
});

for (const command of ['start', 'restart']) test(`pending refactor decision warns without blocking ${command} or resolving the decision`, t => {
  const { root, run } = setup(t);
  writeFileSync(join(root, '.reuse/decisions.json'), JSON.stringify({ version: 1, requests: [{ id: 'existing', status: 'pending', kind: 'index', summary: 'Unclear contract', options: ['Keep', 'Change'], evidence: [{ path: 'src/a.js', hash: '0'.repeat(64) }] }] }));
  const ledger = readFileSync(join(root, '.reuse/decisions.json'), 'utf8');
  const result = run(command);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.equal(readFileSync(join(root, 'events'), 'utf8'), command === 'restart' ? 'stop\nstart\n' : 'start\n');
  assert.match(result.stdout, /WAITING_FOR_USER/);
  assert.match(result.stdout, /continuing local startup/);
  assert.equal(readFileSync(join(root, '.reuse/decisions.json'), 'utf8'), ledger);
  assert.equal(existsSync(join(root, '.cache/reuse/startup-report.json')), false);
  const check = spawnSync(process.execPath, [resolve('hack/reuse/cli.mjs'), 'check'], { cwd: root, encoding: 'utf8' });
  assert.equal(check.status, 2, check.stdout + check.stderr);
});

for (const target of ['frontend', 'server', 'backend', 'all']) test(`targeted restart limits process and readiness work to ${target}`, t => {
  const { root } = setup(t);
  const result = spawnSync('bash', ['-c', `
    source hack/lens-local.sh
    prepare_start() { echo "prepare $1"; }
    cmd_stop() { echo "stop \${1:-all}"; }
    ensure_database() { echo database; }
    ensure_tls() { :; }
    regenerate_mcp_tools() { echo catalog; }
    resolve_free_port() { echo "$2"; }
    cmd_check() { :; }
    start_one() { echo "start $1"; }
    wait_for_http() { echo "wait $1"; }
    print_initial_admin_credentials() { :; }
    lens_local_main restart "$1"
  `, 'test', target], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  const lines = result.stdout.split('\n');
  const effectiveTarget = target === 'server' ? 'backend' : target;
  assert.ok(lines.includes(`stop ${effectiveTarget}`));
  for (const service of ['frontend', 'server', 'backend']) {
    const selected = service !== 'server' && (effectiveTarget === 'all' || effectiveTarget === service);
    assert.equal(lines.includes(`start ${service}`), selected);
    assert.equal(lines.includes(`wait ${service}`), selected);
  }
  assert.equal(lines.includes('database'), effectiveTarget === 'all' || effectiveTarget === 'backend');
  assert.equal(lines.includes('catalog'), false);
});

test('MCP catalog failure prevents stopping existing services', t => {
  const { root, run } = setup(t);
  const result = run('restart', { overrides: 'regenerate_mcp_tools() { return 1; }' });
  assert.equal(result.status, 1, result.stdout + result.stderr);
  assert.ok(!existsSync(join(root, 'events')));
});

test('unknown restart target fails before touching the environment', t => {
  const { root } = setup(t);
  const result = spawnSync('bash', ['-c', 'source hack/lens-local.sh; prepare_start() { echo unexpected; }; lens_local_main restart invalid'], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 2);
  assert.ok(!result.stdout.includes('unexpected'));
});

test('startup does not parse unrelated unregistered implementations', t => {
  const { put, run } = setup(t);
  put('apps/backend/src/llm_d_lens/unrelated.py', 'invalid Python source !');
  const result = run('restart');
  assert.equal(result.status, 0, result.stdout + result.stderr);
});

test('frontend environment checks do not require a Python venv or Rust', t => {
  const { root } = setup(t);
  mkdirSync(join(root, 'node_modules'));
  const result = spawnSync('bash', ['-c', 'source hack/lens-local.sh; cmd_check frontend'], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.ok(!result.stdout.includes('venv:'));
  assert.ok(!result.stdout.includes('rust:'));
});

test('targeted restart restores the actual ports of running peer services', t => {
  const { root, put } = setup(t);
  put('.dev-pids/backend.port', '8099\n');
  const result = spawnSync('bash', ['-c', `
    source hack/lens-local.sh
    is_running() { [[ "$1" == backend ]]; }
    prepare_start() { :; }
    cmd_stop() { :; }
    start_services() { echo "backend=$BACKEND_PORT"; }
    cmd_restart server
  `], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout, /backend=8099/);
});

test('restart all reloads previously recorded ports', t => {
  const { root, put } = setup(t);
  put('.dev-pids/backend.port', '8099\n');
  put('.dev-pids/frontend.port', '5199\n');
  const result = spawnSync('bash', ['-c', `
    source hack/lens-local.sh
    prepare_start() { :; }
    cmd_stop() { :; }
    start_services() { echo "frontend=$FRONTEND_PORT backend=$BACKEND_PORT"; }
    cmd_restart all
  `], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout, /frontend=5199 backend=8099/);
});

test('targeted restart uses the recorded port for a stopped backend', t => {
  const { root, put } = setup(t);
  put('.dev-pids/backend.port', '8099\n');
  const result = spawnSync('bash', ['-c', `
    source hack/lens-local.sh
    prepare_start() { :; }
    cmd_stop() { :; }
    start_services() { echo "backend=$BACKEND_PORT"; }
    cmd_restart backend
  `], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout, /backend=8099/);
});

test('legacy running services without port state fail before stopping anything', t => {
  const { root } = setup(t);
  const result = spawnSync('bash', ['-c', `
    source hack/lens-local.sh
    is_running() { [[ "$1" == backend ]]; }
    prepare_start() { echo unexpected; }
    cmd_stop() { echo unexpected; }
    cmd_restart server
  `], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 1);
  assert.match(result.stdout + result.stderr, /restart all once/);
  assert.ok(!result.stdout.includes('unexpected'));
});

test('targeted restart refuses to move ports without updating peer proxies', t => {
  const { root } = setup(t);
  const result = spawnSync('bash', ['-c', `
    source hack/lens-local.sh
    ensure_tls() { :; }
    resolve_free_port() { echo 19001; }
    cmd_check() { echo unexpected; }
    start_one() { echo unexpected; }
    start_services frontend
  `], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 1);
  assert.match(result.stdout + result.stderr, /update dependent proxies/);
  assert.ok(!result.stdout.includes('unexpected'));
});

test('targeted restart composes with upstream target-node and kubeconfig options', t => {
  const { root, run } = setup(t);
  const result = run('restart', {
    args: ['server', '--target-node', 'node-a', '--kubeconfig', 'cluster.yaml'],
    overrides: 'cmd_stop() { echo "stop:$1" >> "$ROOT_DIR/events"; }; start_services() { echo "start:$1" >> "$ROOT_DIR/events"; }',
  });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.equal(readFileSync(join(root, 'events'), 'utf8'), 'stop:backend\nstart:backend\n');
  assert.match(readFileSync(join(root, 'kubectl-calls'), 'utf8'), /get node node-a/);
  assert.equal(run('restart', { args: ['frontend', 'backend'] }).status, 2);
  assert.equal(run('start', { args: ['frontend'] }).status, 2);
});

test('manual full validation runs every check and propagates failures without audit records', t => {
  const root = mkdtempSync(join(tmpdir(), 'reuse-full-check-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, 'bin'));
  const stub = join(root, 'bin/stub');
  writeFileSync(stub, '#!/bin/sh\necho "$*" >> "$CHECK_CALLS"\nif [ "$FAIL_CHECK" = 1 ] && [ "$1 $2" = "run lint" ]; then exit 3; fi\n');
  chmodSync(stub, 0o755);
  for (const name of ['npm', 'node', 'python3', 'mock-make']) symlinkSync(stub, join(root, 'bin', name));
  for (const failure of ['0', '1']) {
    const calls = join(root, `calls-${failure}`);
    const result = spawnSync('make', ['-f', resolve('Makefile'), 'verify-full', `MAKE=${join(root, 'bin/mock-make')}`], {
      cwd: root, encoding: 'utf8', env: { ...process.env, PATH: `${join(root, 'bin')}:${process.env.PATH}`,
        PRISM_PYTHON_BIN: join(root, 'bin/python3'), CHECK_CALLS: calls, FAIL_CHECK: failure },
    });
    assert.equal(result.status === 0, failure === '0', result.stdout + result.stderr);
    assert.equal(readFileSync(calls, 'utf8').trim().split('\n').length, 15);
    assert.match(result.stdout, /PASS docs-build/);
    if (failure === '1') assert.match(result.stdout, /FAIL lint/);
    assert.equal(existsSync(join(root, '.cache/reuse/tasks')), false);
  }
});
