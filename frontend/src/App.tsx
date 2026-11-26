import { useEffect, useState } from "react";
import { NOTICE, Twin, useTwinState } from "./backend";
import { Calibrate } from "./modules/Calibrate";
import { Generate } from "./modules/Generate";
import { Monitor } from "./modules/Monitor";
import { Morphology } from "./modules/Morphology";
import { C } from "./ui/Plot";
import { Led } from "./ui/controls";

const MODULES = [
  { id: "monitor", code: "01", title: "Monitor", desc: "Live twin, rhythm, state space", el: Monitor },
  { id: "morphology", code: "02", title: "Morphology", desc: "15 wave parameters", el: Morphology },
  { id: "calibrate", code: "03", title: "Calibrate", desc: "Fit to a beat, what-if", el: Calibrate },
  { id: "generate", code: "04", title: "Generate", desc: "DCGAN vs SimDCGAN", el: Generate },
] as const;

function useHashRoute() {
  const read = () => MODULES.find((m) => `#${m.id}` === window.location.hash)?.id ?? "monitor";
  const [id, setId] = useState(read);
  useEffect(() => {
    const on = () => setId(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [id, (v: string) => (window.location.hash = v)] as const;
}

function Clock() {
  const [t, setT] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setT(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return <>{t.toISOString().slice(11, 19)} UTC</>;
}

function Mark() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
      <rect x="0.5" y="0.5" width="31" height="31" fill="none" stroke="#31404a" />
      <path d="M3 18h7l3-9 5 16 3-7h8" fill="none" stroke={C.twin} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

export function App() {
  const twin = useTwinState();
  const [route, go] = useHashRoute();
  const mod = MODULES.find((m) => m.id === route)!;
  const Active = mod.el;
  const server = twin.link === "online" ? "on" : twin.link === "offline" ? "off" : "idle";
  const bundle = twin.status?.bundle ? "on" : twin.link === "online" ? "warn" : "idle";
  const nModels = twin.status?.models.length ?? 0;

  return (
    <Twin.Provider value={twin}>
      <div className="shell">
        <header className="topbar">
          <div className="brand">
            <Mark />
            <div>
              <div className="brand-name">SIMHEART</div>
              <div className="brand-sub">digital twin console</div>
            </div>
          </div>
          <div className="crumb">
            <span>{mod.code}</span>/<b>{mod.title}</b>
          </div>
          <div className="status-strip">
            <div className="stat opt">
              <span className="stat-k">engine</span>
              <span className="stat-v">
                <Led state="on" pulse /> ECGSYN · browser
              </span>
            </div>
            <div className="stat">
              <span className="stat-k">server</span>
              <span className="stat-v">
                <Led state={server} pulse={twin.link === "connecting"} /> {twin.link}
              </span>
            </div>
            <div className="stat">
              <span className="stat-k">model bundle</span>
              <span className="stat-v" title={twin.status?.model_dir}>
                <Led state={bundle} /> {twin.status?.bundle ? `${nModels} generators` : twin.link === "online" ? "not found" : "—"}
              </span>
            </div>
            <div className="stat opt">
              <span className="stat-k">clock</span>
              <span className="stat-v">
                <Clock />
              </span>
            </div>
          </div>
        </header>

        <nav className="rail">
          <div className="rail-label">Modules</div>
          {MODULES.map((m) => (
            <button key={m.id} className={`nav ${m.id === route ? "active" : ""}`} onClick={() => go(m.id)}>
              <span className="nav-code">{m.code}</span>
              <span className="nav-title">{m.title}</span>
              <span className="nav-desc">{m.desc}</span>
            </button>
          ))}
          <div className="rail-foot">
            <div className="rail-label" style={{ padding: "0 0 4px" }}>
              Signal key
            </div>
            {[
              [C.twin, "Twin / simulator"],
              [C.real, "Real (MIT-BIH)"],
              [C.simgan, "SimDCGAN"],
              [C.dcgan, "DCGAN"],
            ].map(([c, l]) => (
              <div className="legend-row" key={l}>
                <span className="swatch" style={{ background: c }} />
                {l}
              </div>
            ))}
            <div style={{ marginTop: 8, color: "var(--text-4)" }}>216 samples · 360 Hz · −200…+397 ms</div>
          </div>
        </nav>

        <main className="main">
          <Active />
        </main>

        <footer className="footer">
          <span className="flag">RESEARCH USE</span>
          <span className="grow">{twin.status?.notice ?? NOTICE}</span>
          <span>MIT-BIH · PhysioNet · ODC-By</span>
        </footer>
      </div>
    </Twin.Provider>
  );
}
