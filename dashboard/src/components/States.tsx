import type { ReactNode } from "react";
import { ApiError } from "../api/client";

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="px-4 py-8 text-sm text-slate-600">
      <span className="mr-2 inline-block h-3 w-3 animate-pulse rounded-full bg-slate-400 align-middle" aria-hidden />
      {label}
    </div>
  );
}

export function ErrorState({ error, title = "Could not load data" }: { error: unknown; title?: string }) {
  const message =
    error instanceof ApiError || error instanceof Error ? error.message : String(error);
  const status = error instanceof ApiError ? ` (HTTP ${error.status})` : "";
  return (
    <div role="alert" className="my-4 rounded border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
      <p className="font-semibold">
        <span aria-hidden>✗ </span>
        {title}
        {status}
      </p>
      <p className="mt-1 font-mono text-xs break-words whitespace-pre-wrap">{message}</p>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="my-4 rounded border border-dashed border-slate-300 bg-white px-4 py-6 text-sm text-slate-700">
      <p className="font-semibold text-slate-800">{title}</p>
      {children ? <div className="mt-1">{children}</div> : null}
    </div>
  );
}

export function Code({ children }: { children: ReactNode }) {
  return <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[0.85em] text-slate-800">{children}</code>;
}
