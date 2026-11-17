"use strict";
const $ = (id) => document.getElementById(id);
const titles = [
  "¿Quién gana?",
  "Cuándo se decide",
  "Repetición",
  "Objetivos",
  "Calculadora",
  "Partida en vivo",
  "Qué aprendimos",
];
const roles = ["Superior", "Jungla", "Central", "Inferior", "Apoyo"];
let view = 0,
  examples = [],
  votes = [0, 0],
  replay = null,
  position = 0,
  replayChart,
  accuracyChart,
  replayIndex = -1,
  generation = 0;
async function api(path, options) {
  const response = await fetch(`/api/${path}`, options);
  const data = await response.json();
  if (!response.ok)
    throw new Error(data.mensaje || "No se pudo cargar. Vuelve a intentar.");
  return data;
}
function report(error) {
  $("status").textContent = error.message;
}
function safe(action) {
  return (...args) =>
    Promise.resolve()
      .then(() => action(...args))
      .catch(report);
}
function node(tag, text, className) {
  const el = document.createElement(tag);
  el.textContent = text;
  if (className) el.className = className;
  return el;
}
function percent(p) {
  return p == null ? "Sin predicción" : `${Math.round(p * 100)} %`;
}
function show(index) {
  view = (index + 7) % 7;
  document.querySelectorAll("main section").forEach((s, i) => {
    s.hidden = i !== view;
  });
  document.querySelectorAll("[data-view]").forEach((b) => {
    if (+b.dataset.view === view) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  $("location").textContent = titles[view];
  $("status").textContent = "";
  if (view === 2) loadReplay().catch(report);
  accuracyChart?.resize();
  replayChart?.resize();
}
function next() {
  show(
    document.body.classList.contains("presenting") ? (view + 1) % 4 : view + 1,
  );
}
function renderTeams() {
  $("teams").replaceChildren();
  $("prediction-result").textContent = "";
  $("winner-result").textContent = "";
  votes = [0, 0];
  const example = examples[+$("example").value];
  if (!example) {
    $("teams").textContent =
      "No hay ejemplos guardados. Restaura la carpeta demo.";
    return;
  }
  ["azul", "rojo"].forEach((team, t) => {
    const article = node("article", "", `team ${t ? "red" : "blue"}`);
    article.append(node("h2", `Equipo ${team}`));
    const list = node("ul", "", "roster");
    example.campeones[team].forEach((name, i) => {
      const li = node("li", name);
      li.append(node("span", roles[i], "role"));
      list.append(li);
    });
    const vote = node("div", "", "vote"),
      button = node("button", `Votar ${team}`),
      count = node("strong", "0");
    count.setAttribute("aria-live", "polite");
    count.setAttribute("aria-label", `Votos del equipo ${team}`);
    button.onclick = () => {
      count.textContent = ++votes[t];
    };
    vote.append(button, count, node("span", "votos"));
    const history = node("dl", "", "team-history");
    const summary = example.historial?.[team] || {};
    [
      ["winrate_10", "Victorias recientes", percent],
      ["kda_medio", "KDA reciente", (v) => v.toFixed(1)],
      ["partidas_con_campeon", "Partidas con el campeón", (v) => v.toFixed(1)],
    ].forEach(([field, label, format]) => {
      const item = node("div");
      item.append(node("dt", label), node("dd", summary[field] == null ? "Sin datos" : format(summary[field])));
      history.append(item);
    });
    article.append(list, history, vote);
    $("teams").append(article);
  });
}
const chartFont = () => document.body.classList.contains("presenting") ? 24 : 16;
function refreshCharts() {
  for (const chart of [accuracyChart, replayChart]) {
    if (!chart) continue;
    chart.options.plugins.legend.labels.font.size = chartFont();
    chart.options.scales.x.ticks.font.size = chartFont();
    chart.options.scales.y.ticks.font.size = chartFont();
    chart.options.scales.x.title.font.size = chartFont();
    chart.update("none");
    chart.resize();
  }
}
const chartOptions = () => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: { legend: { labels: { color: "#202b2e", font: { size: chartFont() } } } },
  scales: {
    x: {
      ticks: { color: "#202b2e", font: { size: chartFont() } },
      title: { display: true, text: "Minuto", font: { size: chartFont() } },
    },
    y: {
      min: 0,
      max: 1,
      ticks: {
        color: "#202b2e",
        callback: (v) => `${Math.round(v * 100)} %`,
        font: { size: chartFont() },
      },
    },
  },
});
async function loadReplay(force = false) {
  const index = +$("replay-example").value;
  if (!examples.length || (!force && index === replayIndex)) return;
  const ticket = ++generation;
  const data = await api(`examples/${index}/replay`);
  if (ticket !== generation) return;
  replay = data;
  replayIndex = index;
  position = 0;
  drawReplay();
}
function drawReplay() {
  if (!replay?.curva.length) {
    $("pause").textContent = "No hay cuadros disponibles.";
    return;
  }
  const points = replay.curva.slice(0, position + 1),
    point = points.at(-1);
  $("minute").textContent = point.minuto;
  $("probability").textContent = percent(point.prob_azul);
  $("pause").textContent =
    point.minuto === 15
      ? "Pausa. ¿Quién gana ahora?"
      : position === replay.curva.length - 1
        ? "Fin de la repetición. El ganador se revela en ¿Quién gana?"
        : "";
  $("step").disabled = position >= replay.curva.length - 1;
  const visibleEvents = replay.eventos.filter((e) => e.minuto <= point.minuto);
  $("events").textContent =
    visibleEvents
      .slice(-4)
      .map((e) => `${e.tipo} ${e.equipo} · ${Math.floor(e.minuto)} min`)
      .join(" / ") || "Todavía no hay objetivos registrados.";
  replayChart?.destroy();
  replayChart = new Chart($("replay"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "Probabilidad del azul",
          data: points.map((p) => ({ x: p.minuto, y: p.prob_azul })),
          borderColor: "#1555a0",
          borderWidth: 3,
          pointRadius: 2,
        },
        ...["dragón", "torre", "barón", "inhibidor"].map((type, i) => ({
          type: "scatter",
          label: type,
          data: visibleEvents
            .filter((e) => e.tipo === type)
            .map((e) => ({
              x: e.minuto,
              y: points.reduce(
                (p, c) => (c.minuto <= e.minuto ? c : p),
                points[0],
              ).prob_azul,
            })),
          backgroundColor: ["#a63231", "#202b2e", "#6f4c89", "#93651d"][i],
          pointStyle: ["triangle", "rect", "rectRot", "star"][i],
          pointRadius: 7,
        })),
      ],
    },
    options: {
      ...chartOptions(),
      scales: {
        ...chartOptions().scales,
        x: {
          type: "linear",
          min: 0,
          max: replay.curva.at(-1).minuto,
          title: { display: true, text: "Minuto" },
          ticks: { font: { size: 16 } },
        },
      },
    },
  });
}
function step() {
  if (replay && position < replay.curva.length - 1) {
    position++;
    drawReplay();
  }
}
async function presentation() {
  const active = document.body.classList.toggle("presenting");
  $("present").textContent = active ? "Salir de presentación" : "Presentar P";
  if (active) {
    if (view > 3) show(0);
    try {
      await document.documentElement.requestFullscreen();
    } catch (_) {
      /* The large layout remains available when fullscreen is denied. */
    }
  } else if (document.fullscreenElement) await document.exitFullscreen();
  refreshCharts();
}
document
  .querySelectorAll("[data-view]")
  .forEach((b) => (b.onclick = () => show(+b.dataset.view)));
$("present").onclick = safe(presentation);
$("next").onclick = next;
$("previous").onclick = () =>
  show(
    document.body.classList.contains("presenting") ? (view + 3) % 4 : view - 1,
  );
$("example").onchange = renderTeams;
$("reset-votes").onclick = renderTeams;
$("prediction").onclick = safe(async () => {
  const index = $("example").value;
  const data = await api(`examples/${index}/prediction`);
  if (index === $("example").value)
    $("prediction-result").replaceChildren(
      node("span", "Modelo antes de jugar: "),
      node("strong", percent(data.prob_azul)),
      node("span", " para el azul"),
    );
});
$("winner").onclick = safe(async () => {
  const index = $("example").value;
  const data = await api(`examples/${index}/winner`);
  if (index === $("example").value)
    $("winner-result").textContent =
      `Ganó el equipo ${data.gana_azul ? "azul" : "rojo"}`;
});
$("replay-example").onchange = safe(() => loadReplay(true));
$("step").onclick = step;
$("replay-reset").onclick = () => {
  position = 0;
  drawReplay();
};
document.addEventListener(
  "keydown",
  safe(async (e) => {
    if (e.target.matches("input,select,textarea") && e.key !== "Escape") return;
    if (e.key.toLowerCase() === "p") {
      e.preventDefault();
      await presentation();
    }
    if (e.key === "ArrowRight") {
      e.preventDefault();
      if (view === 2 && replay && position < replay.curva.length - 1) step();
      else next();
    }
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      if (view === 2 && position > 0) {
        position--;
        drawReplay();
      } else $("previous").click();
    }
    if (e.key.toLowerCase() === "r") {
      const data = await api("comeback");
      $("replay-example").value = data.indice;
      show(2);
      await loadReplay(true);
    }
  }),
);
document.addEventListener("fullscreenchange", () => {
  if (
    !document.fullscreenElement &&
    document.body.classList.contains("presenting")
  ) {
    document.body.classList.remove("presenting");
    $("present").textContent = "Presentar P";
    refreshCharts();
  }
});
$("calculator").onsubmit = safe(async (e) => {
  e.preventDefault();
  const button = e.submitter;
  button.disabled = true;
  $("calculation").textContent = "Calculando…";
  try {
    const values = Object.fromEntries(
      [...$("fields").querySelectorAll("input")].map((i) => [
        i.name,
        Number(i.value),
      ]),
    );
    const data = await api("calculator", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    $("calculation").textContent = `${percent(data.prob_azul)} para el azul`;
    $("factors").replaceChildren(
      ...data.factores.map((f) =>
        node("li", `${f.frase} Aporte al cálculo: ${f.aporte.toFixed(2)}.`),
      ),
    );
  } catch (error) {
    $("calculation").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
$("live-form").onsubmit = safe(async (e) => {
  e.preventDefault();
  e.submitter.disabled = true;
  $("live-result").textContent =
    "Consultando la partida. Puede tardar varios minutos…";
  try {
    const data = await api("live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jugador: $("player").value }),
    });
    $("live-result").textContent =
      `${percent(data.prob_azul)} para el azul. ${data.confianza} ${data.factores.map((f) => f.frase).join(" ")}`;
  } catch (error) {
    $("live-result").textContent = error.message;
  } finally {
    $("player").value = "";
    e.submitter.disabled = false;
  }
});
async function init() {
  const results = await Promise.allSettled([
    api("examples"),
    api("findings"),
    api("calculator"),
    api("live"),
  ]);
  results
    .filter((r) => r.status === "rejected")
    .forEach((r) => report(r.reason));
  if (results[0].status === "fulfilled") {
    examples = results[0].value;
    for (const id of ["example", "replay-example"])
      $("" + id).replaceChildren(
        ...examples.map((e, i) => {
          const o = node("option", e.etiqueta);
          o.value = i;
          return o;
        }),
      );
    if (examples.length === 12) $("example").value = 7;
    renderTeams();
  }
  if (results[1].status === "fulfilled") {
    const data = results[1].value,
      rows = data.predictibilidad.filter(
        (r) => r.minuto >= 5 && r.minuto <= 25 && r.minuto % 5 === 0,
      );
    accuracyChart = new Chart($("accuracy"), {
      type: "line",
      data: {
        labels: [5, 10, 15, 20, 25],
        datasets: [
          ["estado", "Estado de juego", "#1555a0"],
          [
            "previo_misma_poblacion",
            "Previo en las mismas partidas",
            "#a63231",
          ],
        ].map(([model, label, color]) => ({
          label,
          data: rows.filter((r) => r.modelo === model).map((r) => r.acierto),
          borderColor: color,
          borderWidth: 3,
          pointRadius: 4,
        })),
      },
      options: chartOptions(),
    });
    rows.filter((r) => r.modelo === "estado").forEach((r) => {
      const point = node("span");
      point.append(node("span", `${r.minuto} min`), node("strong", percent(r.acierto)),
        node("span", `n = ${r.n.toLocaleString("es-CO")}`));
      $("accuracy-summary").append(point);
    });
    const definitions = [
      ["champion", "Primera sangre"],
      ["dragon", "Primer dragón"],
      ["tower", "Primera torre"],
      ["sangre_torre_dragon", "Las tres juntas"],
      ["baron", "Primer barón"],
      ["inhibitor", "Primer inhibidor"],
    ];
    definitions.forEach(([key, label]) => {
      const r = data.objetivos.find((r) => r.objetivo === key);
      if (!r) return;
      const row = node("div", "", "objective"),
        meter = document.createElement("meter");
      meter.min = 0;
      meter.max = 100;
      meter.value = r.porcentaje_victoria;
      meter.setAttribute("aria-label", label);
      row.append(
        node("span", label),
        meter,
        node("strong", `${Math.round(r.porcentaje_victoria)} %`),
        node("small", `n = ${r.n.toLocaleString("es-CO")}`),
      );
      $("objectives").append(row);
    });
  }
  if (results[2].status === "fulfilled")
    results[2].value.variables.forEach((f) => {
      const label = node("label", f.etiqueta),
        input = document.createElement("input");
      input.type = "number";
      input.name = f.nombre;
      input.value = f.valor;
      input.step = "any";
      input.min = -10000;
      input.max = 10000;
      input.required = true;
      label.append(input);
      $("fields").append(label);
    });
  if (results[3].status === "fulfilled") {
    $("live-status").textContent = results[3].value.mensaje;
    $("live-form").hidden = !results[3].value.disponible;
  }
}
init().catch(report);
