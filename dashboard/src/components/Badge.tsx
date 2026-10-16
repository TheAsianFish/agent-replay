import type { ReactNode } from "react";
import type { DiffOp } from "../api/types";

type Tone = "green" | "red" | "amber" | "blue" | "violet" | "slate" | "cyan";

const TONES: Record<Tone, string> = {
  green: "border-green-300 bg-green-50 text-green-800",
  red: "border-red-300 bg-red-50 text-red-800",
  amber: "border-amber-300 bg-amber-50 text-amber-900",
  blue: "border-blue-300 bg-blue-50 text-blue-800",
  violet: "border-violet-300 bg-violet-50 text-violet-800",
  cyan: "border-cyan-300 bg-cyan-50 text-cyan-900",
  slate: "border-slate-300 bg-slate-100 text-slate-700",
};

export function Badge({ tone, icon, children, title }: { tone: Tone; icon?: string; children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-px text-xs font-medium whitespace-nowrap ${TONES[tone]}`}
    >
      {icon ? <span aria-hidden>{icon}</span> : null}
      {children}
    </span>
  );
}

export function ResultBadge({ passed }: { passed: boolean | null | undefined }) {
  if (passed === null || passed === undefined) return <Badge tone="slate" icon="…">running</Badge>;
  return passed ? <Badge tone="green" icon="✓">pass</Badge> : <Badge tone="red" icon="✗">fail</Badge>;
}

const STEP_STATUS: Record<string, { tone: Tone; icon: string; title: string }> = {
  exact: { tone: "green", icon: "=", title: "served from the cassette on an exact match" },
  approximate: { tone: "amber", icon: "≈", title: "served from the cassette on an approximate match" },
  live: { tone: "cyan", icon: "●", title: "called live" },
  unrecorded: { tone: "red", icon: "✗", title: "no recording matched this call" },
  fallback: { tone: "violet", icon: "↪", title: "fell back to a live call" },
  synth: { tone: "violet", icon: "◇", title: "synthesised" },
};

export function StepStatusBadge({ status }: { status: string }) {
  const style = STEP_STATUS[status] ?? { tone: "slate" as const, icon: "?", title: status };
  return (
    <Badge tone={style.tone} icon={style.icon} title={style.title}>
      {status}
    </Badge>
  );
}

export function KindBadge({ kind }: { kind: string }) {
  const icon = kind === "model" ? "◆" : kind === "tool" ? "⚙" : "·";
  return (
    <Badge tone={kind === "model" ? "blue" : "slate"} icon={icon}>
      {kind}
    </Badge>
  );
}

const CLASSIFICATION: Record<string, { tone: Tone; icon: string }> = {
  passing: { tone: "green", icon: "✓" },
  flaky: { tone: "amber", icon: "~" },
  failing: { tone: "red", icon: "✗" },
  broken: { tone: "red", icon: "!" },
};

export function ClassificationBadge({ value }: { value: string }) {
  const style = CLASSIFICATION[value] ?? { tone: "slate" as const, icon: "?" };
  return (
    <Badge tone={style.tone} icon={style.icon}>
      {value}
    </Badge>
  );
}

export const OP_STYLE: Record<DiffOp, { tone: Tone; icon: string; row: string }> = {
  same: { tone: "slate", icon: "=", row: "text-slate-500" },
  changed: { tone: "amber", icon: "~", row: "bg-amber-50" },
  added: { tone: "green", icon: "+", row: "bg-green-50" },
  removed: { tone: "red", icon: "−", row: "bg-red-50" },
};

export function OpBadge({ op }: { op: DiffOp }) {
  const style = OP_STYLE[op];
  return (
    <Badge tone={style.tone} icon={style.icon}>
      {op}
    </Badge>
  );
}

export function GateBadge({ gating, blocked = false }: { gating: boolean | undefined; blocked?: boolean }) {
  return gating ? (
    <Badge tone="green" icon="●" title="ci_low reaches min_kappa: this criterion can fail a build">
      gating
    </Badge>
  ) : (
    <Badge
      tone="slate"
      icon="○"
      title={blocked ? "this report cannot gate whatever its numbers: verdicts are advisory" : "ci_low is below min_kappa (or kappa is undefined): verdicts are advisory"}
    >
      advisory
    </Badge>
  );
}

export function CheckBadge({ passed, advisory }: { passed: boolean; advisory: boolean }) {
  if (passed) return <Badge tone="green" icon="✓">passed</Badge>;
  if (advisory) return <Badge tone="amber" icon="!">advisory fail</Badge>;
  return <Badge tone="red" icon="✗">failed</Badge>;
}
