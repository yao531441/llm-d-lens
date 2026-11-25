import React, { useEffect, useRef, useState } from 'react';
import { Database, Webcam, Rocket, Gauge, Network, HardDrive, Download, Plug, Bot, ChevronDown,
    Users, UserCog, ShieldCheck, KeyRound, Monitor, ScrollText, Boxes, Activity, Waypoints,
    Sparkles, Fingerprint, PanelLeftClose, PanelLeftOpen, SlidersHorizontal,
    UsersRound
} from 'lucide-react';
import { cn } from '../../shared/lib/cn';
import { Button } from '../../shared/ui/Button';
import { HelpTooltip } from '../../shared/ui/FormControls';
import { SimulationIcon } from '../../pages/simulation';
import { ChangePasswordModal, useAuth } from '../../pages/auth';
import { AccessTokensModal } from '../../pages/auth/AccessTokensModal';
import { NAVIGATION_GROUPS } from '../routes/views';

const MENU_ICONS = {
    Database, Webcam, Rocket, Gauge, Network, HardDrive, Download, Plug, Bot,
    Users, UserCog, ShieldCheck, KeyRound, Monitor, ScrollText, Boxes, Activity,
    Waypoints, SimulationIcon, Sparkles, Fingerprint,
};

const ADMIN_SECTION_KEY = 'lens_nav_admin_section';
const DEFAULT_ADMIN_SECTION = 'Platform Management';
const ADMIN_SECTION_META = {
    'Platform Management': { label: 'Platform', icon: SlidersHorizontal },
    'Access Control': { label: 'Access', icon: UsersRound },
    'Security & Audit': { label: 'Security', icon: ShieldCheck },
};

const ITEM_THEMES = {
    'optimization-deployments': {
        activeBg: 'bg-gradient-to-r from-cyan-50 via-sky-50/70 to-white border-cyan-300/60 text-cyan-800 dark:from-cyan-950/20 dark:via-sky-950/10 dark:to-slate-950/20 dark:border-cyan-500/20 dark:text-cyan-300',
        activeIcon: 'bg-gradient-to-br from-cyan-500 to-sky-500 text-white shadow-[0_0_15px_rgba(6,182,212,0.35)]',
        indicator: 'bg-gradient-to-b from-cyan-400 to-sky-500 shadow-[0_0_8px_rgba(6,182,212,0.6)]'
    }
};

export default function LeftNavigation({ currentView, onNavigate, isMobileOpen, isExpanded = true, onToggleExpanded }) {
    const { canView, principal, logout } = useAuth();
    const [accountMenuOpen, setAccountMenuOpen] = useState(false);
    const [showChangePassword, setShowChangePassword] = useState(false);
    const [showAccessTokens, setShowAccessTokens] = useState(false);
    const accountMenuRef = useRef(null);
    const activeAdminSection = NAVIGATION_GROUPS.find(group => group.id === 'administration')
        ?.items.find(item => item.view === currentView)?.section || null;
    const [adminSelection, setAdminSelection] = useState(() => ({
        view: currentView,
        section: activeAdminSection || localStorage.getItem(ADMIN_SECTION_KEY) || DEFAULT_ADMIN_SECTION,
    }));
    const expandedAdminSection = adminSelection.view === currentView
        ? adminSelection.section
        : activeAdminSection || adminSelection.section;

    if (adminSelection.view !== currentView) {
        setAdminSelection({
            view: currentView,
            section: activeAdminSection || adminSelection.section,
        });
    }

    useEffect(() => {
        if (!accountMenuOpen) return undefined;
        const closeAccountMenu = (event) => {
            if (!accountMenuRef.current?.contains(event.target)) setAccountMenuOpen(false);
        };
        document.addEventListener('pointerdown', closeAccountMenu);
        return () => document.removeEventListener('pointerdown', closeAccountMenu);
    }, [accountMenuOpen]);

    const toggleAdminSection = (section) => {
        if (!isExpanded) onToggleExpanded?.();
        const next = isExpanded && expandedAdminSection === section ? '' : section;
        setAdminSelection({ view: currentView, section: next });
        if (next) localStorage.setItem(ADMIN_SECTION_KEY, next);
        else localStorage.removeItem(ADMIN_SECTION_KEY);
    };

    const handleItemClick = (view, disabled) => {
        if (!disabled) {
            onNavigate(view);
        }
    };

    const visibleGroups = NAVIGATION_GROUPS
        .map(group => ({
        ...group,
        items: group.items.filter(item => !item.disabled && canView(item.view))
    })).filter(group => group.items.length > 0);
    const assistantItem = visibleGroups.find(group => group.placement === 'bottom')?.items[0];

    const renderNavigationItem = (item, { nested = false } = {}) => {
        const Icon = MENU_ICONS[item.icon];
        const isActive = currentView === item.view;

        return (
            <React.Fragment key={item.id}>
                {item.separator && (
                    <div className="my-2 border-t border-slate-900/60 mx-3" />
                )}
                <button
                    type="button"
                    onClick={() => handleItemClick(item.view, item.disabled)}
                    aria-disabled={item.disabled}
                    aria-current={isActive ? 'page' : undefined}
                    title={!isExpanded ? item.label : undefined}
                    className={cn(
                        'group relative flex items-center gap-3 px-3 py-1.5 rounded-xl transition-[box-shadow,transform] duration-200 text-left font-normal border',
                        nested ? 'ml-8 w-[calc(100%-2rem)]' : 'w-full',
                        isActive && 'lens-navigation-active',
                        isActive
                            ? (ITEM_THEMES[item.view]?.activeBg || 'bg-gradient-to-r from-cyan-50 via-blue-50/70 to-white border-cyan-300/60 text-cyan-800 shadow-[inset_0_1px_1px_rgba(255,255,255,0.7)] dark:from-cyan-950/20 dark:via-blue-950/10 dark:to-slate-950/20 dark:border-cyan-500/20 dark:text-cyan-300 dark:shadow-[inset_0_1px_1px_rgba(255,255,255,0.02)]')
                            : 'border-transparent text-slate-700 hover:bg-slate-100 hover:text-slate-950 dark:text-slate-400 dark:hover:bg-slate-900/30 dark:hover:text-white cursor-pointer'
                    )}
                >
                    {isActive && (
                        <div className={cn('lens-navigation-indicator absolute left-1 top-2.5 bottom-2.5 w-1 rounded-full', ITEM_THEMES[item.view]?.indicator || 'bg-gradient-to-b from-cyan-400 to-blue-500')} />
                    )}
                    {!nested && <div className={cn(
                        'lens-navigation-icon p-1 rounded-lg transition-[box-shadow,transform] duration-200',
                        isActive
                            ? (ITEM_THEMES[item.view]?.activeIcon || 'bg-gradient-to-br from-cyan-500 to-blue-500 text-white shadow-[0_0_15px_rgba(6,182,212,0.35)]')
                            : cn('bg-transparent group-hover:brightness-125', item.iconColor || 'text-slate-400')
                    )}>
                        <Icon className="h-[18px] w-[18px] shrink-0" />
                    </div>}
                    {isExpanded && (
                        <div className="flex flex-1 items-center justify-between truncate">
                            <span className="flex items-center gap-1.5 min-w-0">
                                <span className={cn('text-sm tracking-wide truncate', isActive ? 'lens-navigation-active-label font-medium' : 'font-normal')}>
                                    {item.label}
                                </span>
                                {item.badge && (
                                    <span className="nav-exp-badge text-[9px] text-amber-300 font-mono px-1.5 py-0.5 rounded bg-amber-950/40 border border-amber-800/50 shrink-0 tracking-wider">
                                        {item.badge}
                                    </span>
                                )}
                            </span>
                            {item.disabled && (
                                <span className="text-[9px] text-slate-500 font-mono px-2 py-0.5 rounded bg-slate-950 border border-slate-900/80 shrink-0 tracking-wider">
                                    Coming soon
                                </span>
                            )}
                        </div>
                    )}
                    {!isExpanded && (
                        <div className="absolute left-16 top-1/2 -translate-y-1/2 px-3 py-1.5 bg-white border border-slate-200/80 text-slate-900 dark:bg-slate-900 dark:border-slate-800/60 dark:text-white text-xs font-medium rounded-lg invisible group-hover:visible shadow-xl z-[99999] whitespace-nowrap flex items-center gap-2">
                            {item.label}
                            {item.badge && <span className="text-[10px] text-amber-300 font-mono">({item.badge})</span>}
                            {item.disabled && <span className="text-[10px] text-slate-500 font-mono">(Coming soon)</span>}
                        </div>
                    )}
                </button>
            </React.Fragment>
        );
    };

    return (
        <>
        <aside className={cn(
            'lens-navigation fixed top-0 left-0 h-dvh',
            isMobileOpen ? 'flex' : 'hidden md:flex',
            'flex-col border-r border-theme-border bg-theme-card transition-all duration-300 z-50',
            isExpanded ? 'w-64' : 'w-20'
        )}>
            <div className={cn(
                'flex h-16 shrink-0 items-center border-b border-theme-border',
                isExpanded ? 'flex-row justify-between px-3' : 'flex-col justify-center gap-1 px-2'
            )}>
                <a
                    href="https://llm-d.ai"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex min-w-0 items-center gap-2"
                    aria-label="llm-d Lens"
                >
                    {isExpanded ? (
                        <>
                        <img src="https://llm-d.ai/img/llm-d-logotype-and-icon.png" alt="" className="h-6 w-auto object-contain" />
                        <span className="select-none truncate bg-gradient-to-r from-sky-500 via-cyan-400 to-teal-400 bg-clip-text text-xl font-bold tracking-wide text-transparent">
                            Lens
                        </span>
                        </>
                    ) : (
                        <img src="/lens_logo.png" alt="" className="h-6 w-6 object-contain" />
                    )}
                </a>
                <button
                    type="button"
                    onClick={onToggleExpanded}
                    aria-label={isExpanded ? 'Collapse sidebar' : 'Expand sidebar'}
                    title={isExpanded ? 'Collapse sidebar' : 'Expand sidebar'}
                    className={cn(
                        'flex shrink-0 items-center justify-center rounded-lg text-theme-muted transition-colors hover:bg-theme-card hover:text-theme-text',
                        isExpanded ? 'h-8 w-8' : 'h-6 w-10'
                    )}
                >
                    {isExpanded ? <PanelLeftClose size={15} /> : <PanelLeftOpen size={15} />}
                </button>
            </div>

            {/* Navigation Items */}
            <div className="min-h-0 flex-1 overflow-y-auto overflow-x-visible py-3 flex flex-col gap-2 px-3 no-scrollbar">
                {visibleGroups.filter(group => group.placement !== 'bottom').map((group, gIdx) => {
                    const isAdministration = group.id === 'administration';
                    return (
                    <div key={group.id ?? gIdx} className="flex flex-col gap-0.5">
                        {/* Group Header with Stable Vertical Footprint */}
                        {group.title && (
                            isExpanded ? (
                                <div className="flex h-7 items-center px-3 text-[10px] font-bold uppercase tracking-widest text-theme-muted">
                                    <span className="truncate">{group.title}</span>
                                </div>
                            ) : (
                                <div className="relative flex h-7 items-center justify-center px-3" title={group.title}>
                                    {gIdx > 0 && <div className="absolute left-[10px] h-px w-[36px] bg-theme-border" />}
                                </div>
                            )
                        )}

                        <div id={`nav-group-${group.id}`} className="flex flex-col gap-1">
                        {isAdministration ? Object.entries(ADMIN_SECTION_META).map(([section, meta]) => {
                            const sectionItems = group.items.filter(item => item.section === section);
                            if (!sectionItems.length) return null;
                            const SectionIcon = meta.icon;
                            const isSectionExpanded = expandedAdminSection === section;
                            return (
                                <div key={section}>
                                    <button
                                        type="button"
                                        onClick={() => toggleAdminSection(section)}
                                        aria-expanded={isSectionExpanded}
                                        className={cn(
                                            'flex w-full items-center rounded-xl px-3 py-2 text-sm transition-colors',
                                            isSectionExpanded ? 'bg-theme-card text-theme-text' : 'text-theme-muted hover:bg-theme-card hover:text-theme-text'
                                        )}
                                    >
                                        <SectionIcon className="h-4 w-4 shrink-0" />
                                        {isExpanded && <span className="ml-3 font-medium">{meta.label}</span>}
                                        {isExpanded && <ChevronDown className={cn('ml-auto h-3.5 w-3.5 transition-transform', !isSectionExpanded && '-rotate-90')} />}
                                    </button>
                                    {isSectionExpanded && (
                                        <div className="mt-1 border-l border-theme-border pl-1">
                                            {sectionItems.map(item => renderNavigationItem(item, { nested: true }))}
                                        </div>
                                    )}
                                </div>
                            );
                        }) : group.items.map(item => renderNavigationItem(item))}
                        </div>
                    </div>
                    );
                })}
            </div>

            <div className={cn(
                'lens-navigation-footer relative mt-auto shrink-0 border-t border-theme-border bg-theme-card',
                'flex gap-1.5 p-2',
                isExpanded ? 'flex-row items-center' : 'flex-col'
            )} id="navigation-footer">
                <div className={cn('flex h-10 min-w-0 items-center', isExpanded ? 'flex-1' : 'w-full')}>
                    {principal && (
                        <div ref={accountMenuRef} className="relative z-50 min-w-0 flex-1">
                            <button
                                type="button"
                                onClick={() => setAccountMenuOpen((open) => !open)}
                                className={cn(
                                    'flex h-10 w-full min-w-0 items-center rounded-xl border border-theme-border text-theme-muted transition-colors hover:bg-theme-bg hover:text-theme-text',
                                    isExpanded ? 'gap-2 px-2.5' : 'justify-center'
                                )}
                                title={`Account: ${principal.username}`}
                                aria-label="Account menu"
                                aria-haspopup="menu"
                                aria-expanded={accountMenuOpen}
                            >
                                <span className="sidebar-account-avatar flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-theme-border bg-theme-bg text-xs font-semibold uppercase text-theme-text">
                                    {(principal.username || '?').slice(0, 1)}
                                </span>
                                {isExpanded && (
                                    <>
                                        <span className="flex min-w-0 flex-1 flex-col text-left">
                                            <span className="truncate text-xs font-semibold leading-4">Account</span>
                                            {principal.username && (
                                                <span className="truncate text-[11px] font-normal leading-4">{principal.username}</span>
                                            )}
                                        </span>
                                        <ChevronDown size={14} className={cn('shrink-0 transition-transform', accountMenuOpen && 'rotate-180')} />
                                    </>
                                )}
                            </button>
                            {accountMenuOpen && (
                                <div
                                    role="menu"
                                    className="absolute bottom-full left-0 z-50 mb-1 w-44 overflow-hidden rounded-xl border border-theme-border bg-theme-card shadow-2xl"
                                >
                                    <button
                                        type="button"
                                        role="menuitem"
                                        onClick={() => { setAccountMenuOpen(false); setShowChangePassword(true); }}
                                        className="block w-full px-3 py-2 text-left text-xs text-theme-text transition-colors hover:bg-theme-bg"
                                    >
                                        Change password
                                    </button>
                                    <button type="button" role="menuitem"
                                        onClick={() => { setAccountMenuOpen(false); setShowAccessTokens(true); }}
                                        className="block w-full px-3 py-2 text-left text-xs text-theme-text transition-colors hover:bg-theme-bg">
                                        Personal access tokens
                                    </button>
                                    <button
                                        type="button"
                                        role="menuitem"
                                        onClick={() => { setAccountMenuOpen(false); logout(); }}
                                        className="block w-full border-t border-theme-border px-3 py-2 text-left text-xs text-rose-600 transition-colors hover:bg-theme-bg dark:text-rose-300"
                                    >
                                        Sign out
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div>
                {assistantItem && (
                    <HelpTooltip
                        className={cn('relative z-50 shrink-0', isExpanded ? 'w-10' : 'w-full')}
                        tooltipClassName={cn('bottom-full pb-2 whitespace-nowrap', isExpanded ? 'right-0' : 'left-0')}
                        trigger={(
                            <Button
                                variant={currentView === assistantItem.view ? 'sky' : 'ghost'}
                                size="icon"
                                onClick={() => handleItemClick(assistantItem.view, assistantItem.disabled)}
                                aria-current={currentView === assistantItem.view ? 'page' : undefined}
                                aria-label={assistantItem.label}
                                className="h-10 w-full rounded-xl transition-[scale,box-shadow,background-color,color] duration-200 hover:shadow-md focus-visible:shadow-md motion-safe:hover:scale-110 motion-safe:active:scale-95 motion-reduce:transition-none"
                            >
                                <Bot size={16} aria-hidden="true" />
                            </Button>
                        )}
                    >
                        {assistantItem.label}
                    </HelpTooltip>
                )}
            </div>
        </aside>
        {showChangePassword && (
            <ChangePasswordModal
                onClose={() => setShowChangePassword(false)}
                onSaved={() => setShowChangePassword(false)}
            />
        )}
        {showAccessTokens && <AccessTokensModal onClose={() => setShowAccessTokens(false)} />}
        </>
    );
}
