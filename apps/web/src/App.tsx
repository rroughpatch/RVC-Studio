const surfaces = [
  {
    title: "Browser UI",
    body: "Vite + React becomes the primary shell for uploads, settings, progress, and audio previews.",
  },
  {
    title: "Gateway",
    body: "Elysia owns the local HTTP surface, process orchestration, and the browser-facing API contract.",
  },
  {
    title: "ML Service",
    body: "Python stays responsible for RVC, UVR, TTS, training, and other model-bound audio compute.",
  },
];

const migrationNotes = [
  "Current Python entrypoints still assume the repo root is the working directory.",
  "The first browser-facing API should front inference, uploads, job status, and logs.",
  "Realtime audio and playlist playback need a separate streaming design instead of a direct Streamlit port.",
];

export default function App() {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(255,214,102,0.16),transparent_30%),linear-gradient(180deg,#101726_0%,#0b1020_55%,#05070f_100%)] text-slate-100">
      <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-10 px-6 py-10 md:px-10">
        <section className="grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
          <div className="rounded-[2rem] border border-white/10 bg-white/6 p-8 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur">
            <p className="text-xs font-semibold uppercase tracking-[0.35em] text-amber-200/80">
              RVC Studio
            </p>
            <h1 className="mt-4 max-w-3xl text-4xl font-semibold tracking-tight text-white md:text-6xl">
              Browser-first shell for the Bun and Elysia migration.
            </h1>
            <p className="mt-5 max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
              This workspace now has dedicated app and service boundaries. The UI
              moves into React, the local control plane moves into Elysia, and the
              Python stack is reduced to the model-heavy audio service.
            </p>
            <div className="mt-8 flex flex-wrap gap-3 text-sm text-slate-200">
              <span className="rounded-full border border-amber-300/25 bg-amber-300/10 px-4 py-2">
                apps/web
              </span>
              <span className="rounded-full border border-cyan-300/25 bg-cyan-300/10 px-4 py-2">
                services/gateway
              </span>
              <span className="rounded-full border border-emerald-300/25 bg-emerald-300/10 px-4 py-2">
                services/ml
              </span>
            </div>
          </div>

          <aside className="rounded-[2rem] border border-white/10 bg-slate-950/50 p-6 shadow-[0_12px_40px_rgba(0,0,0,0.3)]">
            <p className="text-sm font-medium uppercase tracking-[0.28em] text-slate-400">
              Next
            </p>
            <ol className="mt-5 space-y-4 text-sm leading-6 text-slate-300">
              <li>Wire the web app to the gateway health and metadata routes.</li>
              <li>Expose file uploads, model lists, and inference jobs through Elysia.</li>
              <li>Move Python compute behind a stable internal service contract.</li>
            </ol>
          </aside>
        </section>

        <section className="grid gap-5 md:grid-cols-3">
          {surfaces.map((surface) => (
            <article
              key={surface.title}
              className="rounded-[1.5rem] border border-white/10 bg-slate-900/45 p-6"
            >
              <h2 className="text-xl font-semibold text-white">{surface.title}</h2>
              <p className="mt-3 text-sm leading-6 text-slate-300">{surface.body}</p>
            </article>
          ))}
        </section>

        <section className="rounded-[2rem] border border-white/10 bg-black/20 p-8">
          <p className="text-sm font-medium uppercase tracking-[0.28em] text-slate-400">
            Migration Constraints
          </p>
          <ul className="mt-5 space-y-3 text-sm leading-6 text-slate-300">
            {migrationNotes.map((note) => (
              <li key={note} className="rounded-2xl border border-white/8 bg-white/4 px-4 py-3">
                {note}
              </li>
            ))}
          </ul>
        </section>
      </main>
    </div>
  );
}
