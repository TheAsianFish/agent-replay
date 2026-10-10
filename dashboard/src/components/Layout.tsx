import { NavLink, Outlet } from "react-router";
import { useHealth } from "../api/queries";

const SECTIONS = [
  { to: "/", label: "Runs", end: true },
  { to: "/diff", label: "Diff", end: false },
  { to: "/suites", label: "Suites", end: false },
  { to: "/flaky", label: "Flakiness", end: false },
  { to: "/calibration", label: "Calibration", end: false },
] as const;

function HealthInfo() {
  const health = useHealth();
  if (health.isPending) return <span className="text-slate-400">connecting…</span>;
  if (health.isError) {
    return (
      <span className="text-red-300" role="alert">
        <span aria-hidden>✗ </span>API unreachable
      </span>
    );
  }
  const { backend, revision, version } = health.data;
  return (
    <span className="num" data-testid="health">
      store <span className="font-mono text-slate-100">{backend}</span> · revision{" "}
      <span className="font-mono text-slate-100">{revision ?? "unversioned"}</span> · v{version}
    </span>
  );
}

export function Layout() {
  return (
    <div className="min-h-screen">
      <header className="bg-slate-900 text-slate-300">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2">
          <NavLink to="/" className="font-mono text-sm font-semibold text-white">
            agent-replay
          </NavLink>
          <nav aria-label="Sections" className="flex flex-wrap gap-1 text-sm">
            {SECTIONS.map((s) => (
              <NavLink
                key={s.to}
                to={s.to}
                end={s.end}
                className={({ isActive }) =>
                  `rounded px-2.5 py-1 ${isActive ? "bg-slate-700 font-semibold text-white underline decoration-2 underline-offset-4" : "hover:bg-slate-800 hover:text-white"}`
                }
              >
                {s.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto text-xs">
            <HealthInfo />
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1400px] px-4 py-5">
        <Outlet />
      </main>
    </div>
  );
}

export function PageHeader({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <h1 className="text-xl font-semibold text-slate-900">{title}</h1>
      {children}
    </div>
  );
}
