import { cors } from "@elysiajs/cors";
import { openapi } from "@elysiajs/openapi";
import { Elysia } from "elysia";
import { dataDirectories } from "@rvc-studio/shared";

const api = new Elysia({ prefix: "/api" })
  .get(
    "/health",
    () => ({
      service: "gateway",
      status: "ok" as const,
      ml: {
        status: "pending-migration" as const,
      },
    }),
    {
      detail: {
        summary: "Check gateway health",
        tags: ["system"],
      },
    },
  )
  .get(
    "/layout",
    () => ({
      apps: ["apps/web"],
      services: ["services/gateway", "services/ml"],
      dataDirectories,
    }),
    {
      detail: {
        summary: "Describe the current monorepo layout",
        tags: ["system"],
      },
    },
  );

const app = new Elysia()
  .use(
    cors({
      origin: true,
    }),
  )
  .use(
    openapi({
      documentation: {
        info: {
          title: "RVC Studio Gateway",
          version: "0.1.0",
          description:
            "Local Bun + Elysia control plane for the browser app migration.",
        },
      },
    }),
  )
  .get("/", () => ({
    service: "rvc-studio-gateway",
    status: "ok" as const,
    docs: "/openapi",
    api: "/api",
  }))
  .use(api);

const port = Number(process.env.GATEWAY_PORT ?? 3000);

app.listen(port);

console.log(`RVC Studio gateway listening on http://127.0.0.1:${port}`);

export type GatewayApp = typeof app;
