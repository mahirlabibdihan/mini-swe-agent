import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("compare", "routes/compare.tsx"),
  route("jobs/:jobName", "routes/job.tsx"),
  route("jobs/:jobName/critiques", "routes/critiques.tsx"),
  route(
    "jobs/:jobName/critiques/:critiqueRunName",
    "routes/critique-run.tsx"
  ),
  route(
    "jobs/:jobName/tasks/:source/:agent/:modelProvider/:modelName/:taskName",
    "routes/task.tsx"
  ),
  route(
    "jobs/:jobName/tasks/:source/:agent/:modelProvider/:modelName/:taskName/trials/:trialName",
    "routes/trial.tsx"
  ),
  route("task-definitions", "routes/task-definitions.tsx"),
  route("task-definitions/:taskName", "routes/task-definition.tsx"),
  route("prototypes/chart-toolbar", "routes/chart-toolbar-prototypes.tsx"),
] satisfies RouteConfig;
