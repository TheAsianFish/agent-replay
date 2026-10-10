import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { client, unwrap } from "./client";

export interface RunFilters {
  limit: number;
  offset: number;
  scenario?: string | undefined;
  suite_id?: string | undefined;
  mode?: string | undefined;
  passed?: boolean | undefined;
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: async () => unwrap(await client.GET("/api/health")),
    staleTime: 60_000,
  });
}

export function useRuns(filters: RunFilters, enabled = true) {
  return useQuery({
    queryKey: ["runs", filters],
    queryFn: async () => unwrap(await client.GET("/api/runs", { params: { query: filters } })),
    placeholderData: keepPreviousData,
    enabled,
  });
}

export function useRun(id: string) {
  return useQuery({
    queryKey: ["run", id],
    queryFn: async () =>
      unwrap(await client.GET("/api/runs/{run_id}", { params: { path: { run_id: id } } })),
  });
}

export function useScenarios() {
  return useQuery({
    queryKey: ["scenarios"],
    queryFn: async () => unwrap(await client.GET("/api/scenarios")),
  });
}

export function useDiff(base: string, head: string) {
  return useQuery({
    queryKey: ["diff", base, head],
    queryFn: async () =>
      unwrap(await client.GET("/api/diff", { params: { query: { base, head } } })),
    enabled: base !== "" && head !== "",
  });
}

export function useSuites() {
  return useQuery({
    queryKey: ["suites"],
    queryFn: async () => unwrap(await client.GET("/api/suites", { params: { query: { limit: 200 } } })),
  });
}

export function useSuite(id: string | undefined) {
  return useQuery({
    queryKey: ["suite", id],
    queryFn: async () =>
      unwrap(
        await client.GET("/api/suites/{suite_id}", { params: { path: { suite_id: id ?? "" } } }),
      ),
    enabled: id !== undefined && id !== "",
  });
}

export function useFlakeReports() {
  return useQuery({
    queryKey: ["flake"],
    queryFn: async () => unwrap(await client.GET("/api/flake", { params: { query: { limit: 200 } } })),
  });
}

export function useFlakeReport(id: string | undefined) {
  return useQuery({
    queryKey: ["flake", id],
    queryFn: async () =>
      unwrap(
        await client.GET("/api/flake/{report_id}", { params: { path: { report_id: id ?? "" } } }),
      ),
    enabled: id !== undefined,
  });
}

export function useFlakeTrend(scenario: string | undefined) {
  return useQuery({
    queryKey: ["flake-trend", scenario],
    queryFn: async () =>
      unwrap(await client.GET("/api/flake/trend", { params: { query: { scenario } } })),
    enabled: scenario !== undefined,
  });
}

export function useCalibrations() {
  return useQuery({
    queryKey: ["calibration"],
    queryFn: async () => unwrap(await client.GET("/api/calibration")),
  });
}

export function useCalibration(id: string | undefined) {
  return useQuery({
    queryKey: ["calibration", id],
    queryFn: async () =>
      unwrap(
        await client.GET("/api/calibration/{report_id}", {
          params: { path: { report_id: id ?? "" } },
        }),
      ),
    enabled: id !== undefined,
  });
}
