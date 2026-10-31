import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

# ======================================================================
# PAGE CONFIGURATION
# ======================================================================
st.set_page_config(
    page_title="Bankruptcy Prevention System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ======================================================================
# DESIGN TOKENS
#   ink     #0B1F33  deep navy (hero, headings)
#   paper   #F5F7FA  page background
#   teal    #0E9F8E  healthy / non-bankruptcy
#   signal  #D64545  distress / bankruptcy
#   brass   #C79A3B  accent (coins, highlights)
# ======================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Libre Franklin', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: #F5F7FA; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

/* ---------- headings ---------- */
h2, h3 { color: #0B1F33; letter-spacing: -0.01em; font-weight: 800; }

/* ---------- KPI strip ---------- */
.kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1.2rem 0 2rem; }
.kpi {
    background: #fff;
    border: 1px solid #DCE3EC;
    border-left: 4px solid #C79A3B;
    border-radius: 6px;
    padding: 16px 18px;
}
.kpi b { display: block; font-size: 1.7rem; color: #0B1F33; font-weight: 800; line-height: 1.1; }
.kpi span { color: #5B6B7F; font-size: 0.86rem; }

/* ---------- factor cards ---------- */
.factor-note { color: #5B6B7F; font-size: 0.9rem; margin-top: -0.3rem; }

/* ---------- result panel ---------- */
.verdict {
    border-radius: 8px;
    padding: 20px 22px;
    color: #fff;
    margin-bottom: 12px;
}
.verdict.risk { background: #D64545; }
.verdict.safe { background: #0E9F8E; }
.verdict h3 { color: #fff; margin: 0 0 4px; font-size: 1.5rem; }
.verdict p { margin: 0; opacity: 0.92; }

.bar-wrap { margin: 10px 0 14px; }
.bar-label { display: flex; justify-content: space-between; font-size: 0.88rem; color: #0B1F33; margin-bottom: 4px; }
.bar-track { background: #E4EAF1; border-radius: 4px; height: 10px; overflow: hidden; }
.bar-fill { height: 10px; border-radius: 4px; }

/* ---------- buttons ---------- */
.stFormSubmitButton button, .stButton button[kind="primary"] {
    width: 100%;
    min-height: 2.9rem;
    border-radius: 6px;
    font-weight: 600;
    background: #0B1F33;
    border: 1px solid #0B1F33;
    color: #fff;
}
.stFormSubmitButton button:hover { background: #17365D; border-color: #17365D; color: #fff; }
.stFormSubmitButton button:focus-visible { outline: 3px solid #C79A3B; outline-offset: 2px; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #DCE3EC; }
.stTabs [data-baseweb="tab"] { font-weight: 600; color: #5B6B7F; }
.stTabs [aria-selected="true"] { color: #0B1F33; }

.disclaimer { color: #6B7A8D; font-size: 0.82rem; border-top: 1px solid #DCE3EC; padding-top: 14px; margin-top: 32px; }

@media (max-width: 800px) {
    .kpi-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================================
# 3D SCENES (Three.js, rendered inside Streamlit components)
# ======================================================================
THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"

HERO_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; font-family: 'Libre Franklin', system-ui, -apple-system, 'Segoe UI', sans-serif; }
  #wrap {
    position: relative; width: 100%; height: 100%;
    background: radial-gradient(120% 120% at 75% 40%, #17365D 0%, #0B1F33 55%, #071624 100%);
    border-radius: 10px; overflow: hidden;
  }
  canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  .copy { position: absolute; left: 44px; top: 50%; transform: translateY(-50%); max-width: 470px; color: #fff; z-index: 2; }
  .copy h1 { font-size: 2.6rem; line-height: 1.08; margin: 0 0 14px; font-weight: 800; letter-spacing: -0.02em; }
  .copy p { margin: 0 0 20px; color: #B8C7D9; font-size: 1.02rem; line-height: 1.55; }
  .tags span { display: inline-block; border: 1px solid rgba(199,154,59,0.55); color: #E8C97E; padding: 5px 11px; border-radius: 4px; font-size: 0.8rem; margin: 0 6px 6px 0; }
  @media (max-width: 760px) {
    .copy { left: 22px; right: 22px; top: 22px; transform: none; max-width: none; }
    .copy h1 { font-size: 1.7rem; }
    .copy p { font-size: 0.9rem; }
  }
  @media (prefers-reduced-motion: reduce) { canvas { opacity: 0.9; } }
</style>
</head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div class="copy">
    <h1>See financial distress before it becomes failure.</h1>
    <p>Enter six qualitative risk factors and get an instant bankruptcy classification from a trained Random Forest model.</p>
    <div class="tags"><span>Random Forest classifier</span><span>6 risk factors</span><span>Instant result</span></div>
  </div>
</div>
<script src="__THREE__"></script>
<script>
(function () {
  const wrap = document.getElementById('wrap');
  const canvas = document.getElementById('c');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0.6, 11);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const l1 = new THREE.PointLight(0x38bdf8, 1.4, 40); l1.position.set(-6, 5, 6); scene.add(l1);
  const l2 = new THREE.PointLight(0xc79a3b, 1.3, 40); l2.position.set(6, -2, 6); scene.add(l2);

  const root = new THREE.Group();
  scene.add(root);

  // Crystal core: wireframe shell + solid faceted gem
  const shell = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.7, 1),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0.55 })
  );
  const gem = new THREE.Mesh(
    new THREE.OctahedronGeometry(1.05, 0),
    new THREE.MeshStandardMaterial({ color: 0x1d4ed8, metalness: 0.7, roughness: 0.25, emissive: 0x0a2a78, flatShading: true })
  );
  root.add(shell); root.add(gem);

  // Orbit rings
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xc79a3b, transparent: true, opacity: 0.7 });
  const ring1 = new THREE.Mesh(new THREE.TorusGeometry(2.7, 0.025, 12, 220), ringMat);
  const ring2 = new THREE.Mesh(new THREE.TorusGeometry(3.2, 0.018, 12, 220), ringMat);
  ring1.rotation.x = Math.PI / 2.6; ring2.rotation.set(Math.PI / 2.1, 0.5, 0);
  root.add(ring1); root.add(ring2);

  // Coins orbiting the core
  const coins = [];
  const coinGeo = new THREE.CylinderGeometry(0.34, 0.34, 0.07, 36);
  const coinMat = new THREE.MeshStandardMaterial({ color: 0xc79a3b, metalness: 0.9, roughness: 0.28, emissive: 0x4a3510 });
  for (let i = 0; i < 6; i++) {
    const m = new THREE.Mesh(coinGeo, coinMat);
    m.userData = { a: (i / 6) * Math.PI * 2, r: i % 2 ? 2.7 : 3.2, tilt: i % 2 ? Math.PI / 2.6 : Math.PI / 2.1 };
    coins.push(m); root.add(m);
  }

  // Balance-sheet skyline: bars at the base
  const bars = [];
  const heights = [0.9, 1.5, 1.1, 2.0, 1.6, 2.5, 2.1, 1.3];
  const skyline = new THREE.Group();
  heights.forEach((h, i) => {
    const color = i === 3 ? 0xd64545 : 0x0e9f8e;
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.34, 1, 0.34),
      new THREE.MeshStandardMaterial({ color, metalness: 0.35, roughness: 0.4, emissive: color, emissiveIntensity: 0.18 })
    );
    mesh.userData = { h, phase: i * 0.8 };
    mesh.position.x = (i - 3.5) * 0.62;
    bars.push(mesh); skyline.add(mesh);
  });
  skyline.position.set(0, -3.3, 0);
  root.add(skyline);

  // Dust particles
  const N = 380, pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const r = 5 + Math.random() * 7, t = Math.random() * Math.PI * 2, p = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(p) * Math.cos(t);
    pos[i * 3 + 1] = r * Math.sin(p) * Math.sin(t) * 0.6;
    pos[i * 3 + 2] = r * Math.cos(p) - 2;
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const points = new THREE.Points(pg, new THREE.PointsMaterial({ color: 0x8fb4d9, size: 0.045, transparent: true, opacity: 0.7 }));
  scene.add(points);

  let mx = 0, my = 0;
  window.addEventListener('pointermove', e => {
    const b = wrap.getBoundingClientRect();
    mx = ((e.clientX - b.left) / b.width - 0.5);
    my = ((e.clientY - b.top) / b.height - 0.5);
  });

  function resize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    const wide = w > 760;
    root.position.set(wide ? 3.1 : 0, wide ? 0 : -1.0, 0);
    root.scale.setScalar(wide ? 1 : 0.7);
  }
  window.addEventListener('resize', resize); resize();

  const clock = new THREE.Clock();
  function frame() {
    const t = reduce ? 0 : clock.getElapsedTime();
    shell.rotation.y = t * 0.25; shell.rotation.x = t * 0.12;
    gem.rotation.y = -t * 0.45; gem.rotation.z = t * 0.2;
    ring1.rotation.z = t * 0.2; ring2.rotation.z = -t * 0.16;
    coins.forEach(c => {
      const u = c.userData, a = u.a + t * (u.r > 3 ? 0.35 : 0.5);
      const x = Math.cos(a) * u.r, y = Math.sin(a) * u.r;
      c.position.set(x, y * Math.cos(u.tilt), y * Math.sin(u.tilt) * 0.6);
      c.rotation.x = Math.PI / 2 + t; c.rotation.z = t * 0.6;
    });
    bars.forEach(b => {
      const s = b.userData.h * (1 + 0.12 * Math.sin(t * 1.2 + b.userData.phase));
      b.scale.y = s; b.position.y = s / 2;
    });
    points.rotation.y = t * 0.02;
    root.rotation.y += ((mx * 0.5) - root.rotation.y) * 0.04;
    root.rotation.x += ((my * 0.25) - root.rotation.x) * 0.04;
    renderer.render(scene, camera);
    if (!reduce) requestAnimationFrame(frame);
  }
  frame();
})();
</script>
</body>
</html>
""".replace("__THREE__", THREE_CDN)


RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; }
  #wrap { width: 100%; height: 100%; border-radius: 10px; overflow: hidden;
          background: radial-gradient(100% 100% at 50% 35%, __BG1__ 0%, __BG2__ 100%); }
  canvas { width: 100%; height: 100%; display: block; }
</style>
</head>
<body>
<div id="wrap"><canvas id="c"></canvas></div>
<script src="__THREE__"></script>
<script>
(function () {
  const MODE = "__MODE__";            // "safe" or "risk"
  const COLOR = __COLOR__;            // main hex
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const wrap = document.getElementById('wrap');
  const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('c'), antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0.5, 9);
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const pl = new THREE.PointLight(0xffffff, 1.5, 40); pl.position.set(4, 5, 6); scene.add(pl);

  const root = new THREE.Group(); scene.add(root);

  const geo = MODE === "safe" ? new THREE.OctahedronGeometry(1.4, 0) : new THREE.IcosahedronGeometry(1.4, 0);
  const core = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color: COLOR, metalness: 0.6, roughness: 0.3, flatShading: true, emissive: COLOR, emissiveIntensity: 0.25 }));
  root.add(core);
  const halo = new THREE.Mesh(new THREE.IcosahedronGeometry(1.9, 1),
    new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.25 }));
  root.add(halo);

  // Bars trend up when safe, down when at risk
  const bars = [];
  const n = 7, grp = new THREE.Group();
  for (let i = 0; i < n; i++) {
    const frac = i / (n - 1);
    const h = MODE === "safe" ? 0.5 + frac * 1.8 : 2.3 - frac * 1.8;
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.3, h, 0.3),
      new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.3, roughness: 0.45, transparent: true, opacity: 0.85 }));
    m.userData = { h, phase: i };
    m.position.x = (i - (n - 1) / 2) * 0.55;
    bars.push(m); grp.add(m);
  }
  grp.position.y = -2.6;
  root.add(grp);

  function resize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize); resize();

  const clock = new THREE.Clock();
  function frame() {
    const t = reduce ? 0 : clock.getElapsedTime();
    core.rotation.y = t * 0.6; core.rotation.x = t * 0.25;
    halo.rotation.y = -t * 0.3;
    if (MODE === "risk") { core.position.x = Math.sin(t * 9) * 0.03; }   // slight tremor
    else { core.position.y = Math.sin(t * 1.4) * 0.12; }
    bars.forEach(b => {
      const k = 1 + 0.06 * Math.sin(t * 1.5 + b.userData.phase);
      b.scale.y = k; b.position.y = (b.userData.h * k) / 2 - b.userData.h / 2;
    });
    renderer.render(scene, camera);
    if (!reduce) requestAnimationFrame(frame);
  }
  frame();
})();
</script>
</body>
</html>
"""


def result_scene(is_risk: bool) -> str:
    return (
        RESULT_HTML.replace("__THREE__", THREE_CDN)
        .replace("__MODE__", "risk" if is_risk else "safe")
        .replace("__COLOR__", "0xD64545" if is_risk else "0x0E9F8E")
        .replace("__BG1__", "#3a1519" if is_risk else "#0f3b3a")
        .replace("__BG2__", "#0B1F33")
    )


# ======================================================================
# LOAD MODEL
# ======================================================================
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "bankruptcy_prevention_model.pkl"
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as error:
    st.error("The model could not be loaded.")
    st.code(str(error))
    st.info(
        "Check that bankruptcy_prevention_model.pkl is in the same "
        "folder as app.py and that the installed scikit-learn version "
        "matches the one used for training."
    )
    st.stop()

# ======================================================================
# FEATURE DEFINITIONS
# ======================================================================
feature_info = {
    "industrial_risk": ("Industrial Risk", "Risk associated with the company's industry."),
    "management_risk": ("Management Risk", "Risk associated with management."),
    "financial_flexibility": ("Financial Flexibility", "Ability to respond to financial challenges."),
    "credibility": ("Credibility", "Credibility-related factor in the dataset."),
    "competitiveness": ("Competitiveness", "Competitiveness-related factor in the dataset."),
    "operating_risk": ("Operating Risk", "Risk associated with business operations."),
}
features = list(feature_info.keys())
LEVELS = [0.0, 0.5, 1.0]

for f in features:
    st.session_state.setdefault(f, 0.0)


def reset_inputs():
    for f in features:
        st.session_state[f] = 0.0
    st.session_state.pop("result", None)


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.title("📊 Bankruptcy Prevention")
    st.caption("Machine Learning Project")
    st.divider()
    st.subheader("How to use")
    st.write("1. Set the six risk factors.")
    st.write("2. Select **Run assessment**.")
    st.write("3. Review the class and confidence.")
    st.write("4. Download the report if you need a record.")
    st.divider()
    st.caption("Model: Random Forest")
    st.caption("Input values: 0, 0.5, or 1")
    st.button("Reset all inputs", on_click=reset_inputs)

# ======================================================================
# HERO (3D)
# ======================================================================
components.html(HERO_HTML, height=430, scrolling=False)

st.markdown(
    """
<div class="kpi-row">
  <div class="kpi"><b>6</b><span>Qualitative risk factors</span></div>
  <div class="kpi"><b>3</b><span>Levels per factor (0, 0.5, 1)</span></div>
  <div class="kpi"><b>2</b><span>Output classes</span></div>
  <div class="kpi"><b>Random Forest</b><span>Ensemble classifier</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ======================================================================
# TABS
# ======================================================================
tab_assess, tab_method, tab_about = st.tabs(["Risk assessment", "How it works", "About the project"])

# ---------------------------- ASSESSMENT ----------------------------
with tab_assess:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.subheader("Company risk factors")
        st.markdown(
            '<div class="factor-note">Choose 0, 0.5 or 1 for every factor, '
            "using the same coding as your training dataset.</div>",
            unsafe_allow_html=True,
        )

        with st.form("risk_assessment_form"):
            c1, c2 = st.columns(2)
            for i, feature in enumerate(features):
                label, help_text = feature_info[feature]
                with (c1 if i % 2 == 0 else c2):
                    st.selectbox(
                        label,
                        options=LEVELS,
                        key=feature,
                        help=help_text,
                        format_func=lambda v: str(v),
                    )
            submitted = st.form_submit_button("Run assessment", type="primary")

        if submitted:
            values = {f: st.session_state[f] for f in features}
            input_df = pd.DataFrame([values], columns=features)
            try:
                prediction = model.predict(input_df)[0]
                proba = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    proba = dict(zip([str(c) for c in model.classes_], [float(p) for p in probs]))
                st.session_state["result"] = {
                    "prediction": str(prediction),
                    "proba": proba,
                    "inputs": values,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            except Exception as error:
                st.session_state.pop("result", None)
                st.error("Prediction failed. Check the inputs and the model file.")
                st.code(str(error))

    with right:
        st.subheader("Assessment result")
        result = st.session_state.get("result")

        if not result:
            st.info("Set the six factors and select **Run assessment** to see the result here.")
        else:
            label = result["prediction"]
            if label == "bankruptcy":
                is_risk = True
                st.markdown(
                    '<div class="verdict risk"><h3>⚠️ Bankruptcy</h3>'
                    "<p>This profile is classified as high risk. "
                    "Consider a review by a qualified financial professional.</p></div>",
                    unsafe_allow_html=True,
                )
            elif label == "non-bankruptcy":
                is_risk = False
                st.markdown(
                    '<div class="verdict safe"><h3>✅ Non-bankruptcy</h3>'
                    "<p>This profile is classified as financially stable.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                is_risk = None
                st.warning(f"The model returned an unexpected class: {label}")

            if is_risk is not None:
                components.html(result_scene(is_risk), height=230, scrolling=False)

            if result["proba"]:
                st.markdown("**Model confidence**")
                for cls, p in sorted(result["proba"].items(), key=lambda kv: -kv[1]):
                    color = "#D64545" if cls == "bankruptcy" else "#0E9F8E"
                    st.markdown(
                        f'<div class="bar-wrap"><div class="bar-label"><span>{cls}</span>'
                        f"<span>{p:.0%}</span></div>"
                        f'<div class="bar-track"><div class="bar-fill" '
                        f'style="width:{p * 100:.1f}%;background:{color}"></div></div></div>',
                        unsafe_allow_html=True,
                    )

            report = pd.DataFrame(
                [{**{feature_info[k][0]: v for k, v in result["inputs"].items()},
                  "Predicted class": label,
                  "Assessed at": result["time"]}]
            )
            with st.expander("View submitted input values"):
                st.dataframe(report, use_container_width=True, hide_index=True)
            st.download_button(
                "Download report (CSV)",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="bankruptcy_assessment.csv",
                mime="text/csv",
            )

# ---------------------------- METHODOLOGY ----------------------------
with tab_method:
    st.subheader("How it works")
    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown("**1. Input**")
        st.write(
            "You provide six qualitative factors: industrial risk, management risk, "
            "financial flexibility, credibility, competitiveness and operating risk."
        )
    with m2:
        st.markdown("**2. Classification**")
        st.write(
            "A Random Forest combines the votes of many decision trees. "
            "Each tree sees the same inputs and the majority class is returned."
        )
    with m3:
        st.markdown("**3. Result**")
        st.write(
            "You get a class (bankruptcy or non-bankruptcy) and, where the model "
            "supports it, the share of trees that voted for each class."
        )
    st.divider()
    st.dataframe(
        pd.DataFrame(
            [{"Factor": v[0], "Description": v[1]} for v in feature_info.values()]
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------- ABOUT ----------------------------
with tab_about:
    st.subheader("About the project")
    st.write(
        "The Bankruptcy Prevention System is a machine-learning project that "
        "classifies a company as bankruptcy or non-bankruptcy from six qualitative "
        "risk factors. It is built with scikit-learn, pandas and Streamlit, and the "
        "3D visuals are rendered with Three.js."
    )
    st.write(
        "Use it as a screening aid. Final decisions should rest on a full "
        "financial analysis."
    )

st.markdown(
    '<div class="disclaimer">Disclaimer: This application provides a machine-learning '
    "estimate based on the supplied dataset. It is not a guarantee of future financial "
    "outcomes or a substitute for professional financial assessment.</div>",
    unsafe_allow_html=True,
)