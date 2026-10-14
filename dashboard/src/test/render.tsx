import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { RouterProvider, createMemoryRouter } from "react-router";
import { vi } from "vitest";
import { routes } from "../routes";
import { health } from "./fixtures";

/** A canned response: JSON with a status (default 200). */
export interface Canned {
  status?: number;
  body: unknown;
}

export type Routes = Record<string, Canned | ((url: URL) => Canned | Promise<Canned>)>;

/**
 * Stub ``fetch``: the first route whose key matches the request path (exact
 * ``/api/runs``, or a prefix ending in ``*``) answers. ``/api/health`` answers
 * by default. Unknown paths get a 404 with a detail, like the server. A route
 * function may return a promise to hold its response back.
 */
export function mockApi(table: Routes) {
  const calls: URL[] = [];
  const all: Routes = { "/api/health": { body: health }, ...table };
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const raw = input instanceof Request ? input.url : input.toString();
    const url = new URL(raw, "http://localhost");
    calls.push(url);
    const key = Object.keys(all).find((k) => (k.endsWith("*") ? url.pathname.startsWith(k.slice(0, -1)) : url.pathname === k));
    const entry = key === undefined ? undefined : all[key];
    const canned: Canned | Promise<Canned> =
      entry === undefined
        ? { status: 404, body: { detail: `no API endpoint ${url.pathname}` } }
        : typeof entry === "function"
          ? entry(url)
          : entry;
    return Promise.resolve(canned).then(
      (c) =>
        new Response(JSON.stringify(c.body), {
          status: c.status ?? 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

/** Render the app at ``path`` with a fresh query cache (no retries). */
export function renderAt(path: string) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  const utils = render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return { ...utils, router };
}
