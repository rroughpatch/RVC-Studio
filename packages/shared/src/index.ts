export const dataDirectories = [
  "configs",
  "datasets",
  "logs",
  "models",
  "output",
  "songs",
] as const;

export const repoPackages = {
  web: "apps/web",
  gateway: "services/gateway",
  ml: "services/ml",
  shared: "packages/shared",
} as const;

export type DataDirectory = (typeof dataDirectories)[number];
