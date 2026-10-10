import createClient from "openapi-fetch";
import type { paths } from "./schema";

/** An API failure; ``message`` is the server's ``detail`` when it sent one. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function detailOf(error: unknown): string | null {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail: unknown = error.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // FastAPI validation errors: [{loc, msg}, ...]
      return detail
        .map((item: unknown) =>
          typeof item === "object" && item !== null && "msg" in item ? String(item.msg) : "",
        )
        .filter(Boolean)
        .join("; ");
    }
  }
  return null;
}

/**
 * The typed client. ``fetch`` is looked up at call time so tests can stub
 * ``globalThis.fetch``.
 */
export const client = createClient<paths>({
  // Same origin as the page; absolute so Request construction works outside browsers too.
  baseUrl: globalThis.location.origin,
  fetch: (request) => globalThis.fetch(request),
});

/** Unwrap an openapi-fetch result, throwing ``ApiError`` with the server's detail. */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  const { data, error, response } = result;
  if (error !== undefined || data === undefined) {
    const detail = detailOf(error);
    throw new ApiError(
      response.status,
      detail ?? `request to ${new URL(response.url || "http://x/").pathname} failed with HTTP ${response.status}`,
    );
  }
  return data;
}
