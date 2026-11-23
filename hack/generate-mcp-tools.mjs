#!/usr/bin/env node
// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const virtualenvPython = path.join(root, '.venv/bin/python');
const python = process.env.PRISM_PYTHON_BIN || (existsSync(virtualenvPython) ? virtualenvPython : 'python3');
const args = process.argv.slice(2);
if (args.some(argument => !['--check', '--json'].includes(argument))) {
    console.error('Usage: node hack/generate-mcp-tools.mjs [--check] [--json]');
    process.exit(2);
}

const result = spawnSync(python, ['-m', 'llm_d_lens.assistant_tools.catalog'], {
    cwd: root,
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
});
if (result.error || result.status !== 0) {
    console.error(result.error?.message || result.stderr || 'Python MCP catalog validation failed.');
    process.exit(result.status || 1);
}
const tools = JSON.parse(result.stdout);
if (args.includes('--json')) {
    process.stdout.write(JSON.stringify(tools, null, 2) + '\n');
} else {
    console.log(`[generate-mcp-tools] validated ${tools.length} Python tools; 0 unexpected skips; no generated Node artifact`);
}