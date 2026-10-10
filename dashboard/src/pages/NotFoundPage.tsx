import { Link, useRouteError } from "react-router";
import { ErrorState } from "../components/States";
import { link } from "../components/table";

export function NotFoundPage() {
  return (
    <div className="py-6">
      <h1 className="text-xl font-semibold">Page not found</h1>
      <p className="mt-2 text-sm">
        No dashboard page at this address. <Link className={link} to="/">Go to runs</Link>.
      </p>
    </div>
  );
}

export function RouteError() {
  const error = useRouteError();
  return (
    <div className="mx-auto max-w-3xl p-6">
      <ErrorState error={error} title="Something went wrong rendering this page" />
      <Link className={link} to="/">Back to runs</Link>
    </div>
  );
}
