/** Auth domain API client (design section 10.1). */

import { postJson, requestJson } from '../../shared/api/httpClient';

export function fetchAuthProviders() {
    return requestJson('/api/v1/auth/providers');
}

export function fetchSession() {
    return requestJson('/api/v1/auth/session');
}

export function login(username, password, remember = false) {
    return postJson('/api/v1/auth/login', { username, password, remember });
}

export function logout() {
    return postJson('/api/v1/auth/logout', {});
}

export function changePassword(oldPassword, newPassword) {
    return postJson('/api/v1/auth/password', { oldPassword, newPassword });
}

export function fetchAccessTokens() {
    return requestJson('/api/v1/auth/access-tokens');
}

export function createAccessToken(name, expiresInDays) {
    return postJson('/api/v1/auth/access-tokens', { name, expiresInDays });
}

export function revokeAccessToken(id) {
    return requestJson(`/api/v1/auth/access-tokens/${encodeURIComponent(id)}`, { method: 'DELETE' });
}
