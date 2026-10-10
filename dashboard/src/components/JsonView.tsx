import { useState } from "react";

/** Strings longer than this are cut with a "show more" toggle. */
export const STRING_LIMIT = 240;

function LongString({ value }: { value: string }) {
  const [open, setOpen] = useState(false);
  const long = value.length > STRING_LIMIT;
  const shown = long && !open ? `${value.slice(0, STRING_LIMIT)}…` : value;
  return (
    <span>
      <span className="break-words whitespace-pre-wrap text-emerald-800">{JSON.stringify(shown)}</span>
      {long ? (
        <button
          type="button"
          onClick={() => {
            setOpen((v) => !v);
          }}
          className="ml-1 rounded px-1 font-sans text-[11px] text-blue-700 underline hover:bg-blue-50"
        >
          {open ? "show less" : `show more (${value.length.toLocaleString("en-US")} chars)`}
        </button>
      ) : null}
    </span>
  );
}

function Scalar({ value }: { value: unknown }) {
  if (value === null) return <span className="text-slate-500">null</span>;
  if (typeof value === "string") return <LongString value={value} />;
  if (typeof value === "number") return <span className="text-blue-800">{value}</span>;
  if (typeof value === "boolean") return <span className="text-violet-800">{String(value)}</span>;
  return <span>{(JSON.stringify(value) as string | undefined) ?? "undefined"}</span>;
}

function Node({ name, value, depth, openDepth }: { name?: string; value: unknown; depth: number; openDepth: number }) {
  const isArray = Array.isArray(value);
  const isObject = typeof value === "object" && value !== null;
  const [open, setOpen] = useState(depth < openDepth);
  const label = name !== undefined ? <span className="text-slate-700">{JSON.stringify(name)}: </span> : null;

  if (!isObject) {
    return (
      <div className="pl-4">
        {label}
        <Scalar value={value} />
      </div>
    );
  }
  const entries: [string, unknown][] = isArray
    ? value.map((v: unknown, i): [string, unknown] => [String(i), v])
    : Object.entries(value);
  const [openBrace, closeBrace] = isArray ? ["[", "]"] : ["{", "}"];
  if (entries.length === 0) {
    return (
      <div className="pl-4">
        {label}
        {openBrace}
        {closeBrace}
      </div>
    );
  }
  return (
    <div className="pl-4">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => {
          setOpen((v) => !v);
        }}
        className="-ml-4 w-4 text-slate-500 hover:text-slate-900"
        aria-label={open ? "collapse" : "expand"}
      >
        {open ? "▾" : "▸"}
      </button>
      {label}
      {openBrace}
      {open ? (
        <>
          {entries.map(([k, v]) => (
            <Node key={k} name={isArray ? undefined : k} value={v} depth={depth + 1} openDepth={openDepth} />
          ))}
          <div>{closeBrace}</div>
        </>
      ) : (
        <span className="text-slate-500">
          {" "}
          {entries.length} {isArray ? "items" : "keys"} {closeBrace}
        </span>
      )}
    </div>
  );
}

/** Pretty-printed, collapsible JSON with long strings truncated. */
export function JsonView({ value, label, openDepth = 2 }: { value: unknown; label?: string; openDepth?: number }) {
  return (
    <section className="min-w-0">
      {label ? <h4 className="mb-1 text-xs font-semibold tracking-wide text-slate-600 uppercase">{label}</h4> : null}
      <div className="max-h-[32rem] overflow-auto rounded border border-slate-200 bg-white py-1 pr-2 font-mono text-xs leading-5">
        <div className="-ml-2">
          <Node value={value} depth={0} openDepth={openDepth} />
        </div>
      </div>
    </section>
  );
}
