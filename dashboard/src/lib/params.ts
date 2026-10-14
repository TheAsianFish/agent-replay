/** A query parameter's value, with an absent or empty value as ``undefined``. */
export function param(params: URLSearchParams, key: string): string | undefined {
  const value = params.get(key);
  return value === null || value === "" ? undefined : value;
}
