import type { RouteObject } from "react-router";
import { Layout } from "./components/Layout";
import { CalibrationDetailPage, CalibrationListPage } from "./pages/CalibrationPage";
import { DiffPage } from "./pages/DiffPage";
import { FlakyPage } from "./pages/FlakyPage";
import { NotFoundPage, RouteError } from "./pages/NotFoundPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { RunsPage } from "./pages/RunsPage";
import { SuitesPage } from "./pages/SuitesPage";

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <RunsPage /> },
      { path: "runs/:id", element: <RunDetailPage /> },
      { path: "diff", element: <DiffPage /> },
      { path: "suites", element: <SuitesPage /> },
      { path: "flaky", element: <FlakyPage /> },
      { path: "calibration", element: <CalibrationListPage /> },
      { path: "calibration/:id", element: <CalibrationDetailPage /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
];
