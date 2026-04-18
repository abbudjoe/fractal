(function () {
    const key = window.FRACTAL_VISUAL_KEY;
    const config = window.FRACTAL_VISUAL_CONFIGS[key];
    if (!config) {
        return;
    }

    const root = document.getElementById("app");
    root.innerHTML = renderShell(config);

    const modeButtons = Array.from(root.querySelectorAll("[data-mode]"));
    const modeTitle = root.querySelector("[data-mode-title]");
    const modeCaption = root.querySelector("[data-mode-caption]");
    const modeInsight = root.querySelector("[data-mode-insight]");
    const metricFocus = root.querySelector("[data-metric-focus]");
    const metricMemory = root.querySelector("[data-metric-memory]");
    const metricCompute = root.querySelector("[data-metric-compute]");
    const canvasHost = root.querySelector("[data-canvas-host]");
    const sceneLabel = root.querySelector("[data-scene-label]");

    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
    renderer.setClearColor(0x000000, 0);
    canvasHost.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
    camera.position.set(0, 7.5, 18);

    const ambient = new THREE.HemisphereLight(0xbed9ff, 0x05070a, 0.95);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 0.95);
    keyLight.position.set(8, 15, 10);
    scene.add(keyLight);

    const rim = new THREE.PointLight(hex(config.accent2), 1.4, 120, 2);
    rim.position.set(-8, 10, -10);
    scene.add(rim);

    const stage = new THREE.Group();
    scene.add(stage);

    const stars = makeStarField();
    scene.add(stars);

    const controls = makeSoftControls(renderer.domElement, stage, camera);
    const runtime = buildScene(config.sceneType, stage, config);

    let currentMode = Object.keys(config.modes)[0];
    applyMode(currentMode);

    modeButtons.forEach((button) => {
        button.addEventListener("click", () => applyMode(button.dataset.mode));
    });

    window.addEventListener("resize", resize);
    resize();

    let start = performance.now();
    function animate(now) {
        requestAnimationFrame(animate);
        const t = (now - start) * 0.001;
        stars.rotation.y = t * 0.01;
        controls.update(t);
        runtime.update(t, currentMode);
        renderer.render(scene, camera);
    }
    requestAnimationFrame(animate);

    function resize() {
        const { clientWidth, clientHeight } = canvasHost;
        renderer.setSize(clientWidth, clientHeight, false);
        camera.aspect = clientWidth / clientHeight;
        camera.updateProjectionMatrix();
    }

    function applyMode(mode) {
        currentMode = mode;
        const modeConfig = config.modes[mode];
        modeButtons.forEach((button) => {
            button.classList.toggle("active", button.dataset.mode === mode);
        });
        modeTitle.textContent = modeConfig.label;
        modeCaption.textContent = modeConfig.caption;
        modeInsight.textContent = modeConfig.insight;
        metricFocus.textContent = modeConfig.metrics.focus;
        metricMemory.textContent = modeConfig.metrics.memory;
        metricCompute.textContent = modeConfig.metrics.compute;
        sceneLabel.textContent = `${config.title} • ${modeConfig.label}`;
        runtime.setMode(mode);
    }
})();

function renderShell(config) {
    const nav = window.FRACTAL_VISUAL_PAGES.map((page) => {
        const active = page.key === config.key ? "active" : "";
        return `<a class="nav-link ${active}" href="${page.href}">${page.label}</a>`;
    }).join("");

    const modeButtons = Object.entries(config.modes)
        .map(
            ([mode, value], index) => `
                <button class="mode-button ${index === 0 ? "active" : ""}" data-mode="${mode}">
                    <strong>${value.label}</strong>
                    <small>${value.short}</small>
                </button>
            `,
        )
        .join("");

    const structure = config.structure
        .map((line) => `<li>${line}</li>`)
        .join("");

    const legend = config.legend
        .map(
            (item) => `
                <div class="legend-row">
                    <span class="legend-swatch" style="color:${item.color}; background:${item.color};"></span>
                    <span>${item.label}</span>
                </div>
            `,
        )
        .join("");

    const facts = config.facts
        .map(
            (item) => `
                <article class="fact-card">
                    <h3>${item.title}</h3>
                    <p>${item.body}</p>
                </article>
            `,
        )
        .join("");

    return `
        <div class="page-shell">
            <header class="hero">
                <div>
                    <div class="eyebrow">
                        <span class="pulse-dot"></span>
                        ${config.tag}
                    </div>
                    <h1 class="title">${config.title}</h1>
                    <p class="subtitle">${config.subtitle}</p>
                </div>
                <nav class="nav-cluster">
                    <a class="nav-link" href="./index.html">Visualization Home</a>
                    ${nav}
                </nav>
            </header>

            <div class="dashboard">
                <aside class="panel sidebar">
                    <section class="section">
                        <h2>Mode</h2>
                        <div class="mode-grid">
                            ${modeButtons}
                        </div>
                    </section>

                    <section class="section">
                        <h2>What You Are Looking At</h2>
                        <ul>${structure}</ul>
                    </section>

                    <section class="section">
                        <h2 data-mode-title></h2>
                        <p data-mode-caption></p>
                    </section>

                    <section class="section">
                        <h2>Key Insight</h2>
                        <p data-mode-insight></p>
                    </section>
                </aside>

                <main class="panel main-stage">
                    <div class="stage-hero">
                        <section class="panel canvas-panel" data-canvas-host>
                            <div class="canvas-overlay">
                                <div class="scene-caption">
                                    <h2 data-scene-label></h2>
                                    <p>${config.sceneHint}</p>
                                </div>
                                <div class="scene-note">Interactive 3D structure + runtime view</div>
                            </div>
                        </section>

                        <aside class="panel hud-card">
                            <div class="metric">
                                <strong>Focus</strong>
                                <span data-metric-focus></span>
                            </div>
                            <div class="metric">
                                <strong>Memory Story</strong>
                                <span data-metric-memory></span>
                            </div>
                            <div class="metric">
                                <strong>Compute Story</strong>
                                <span data-metric-compute></span>
                            </div>
                            <div class="section">
                                <h3>Legend</h3>
                                <div class="legend-grid">
                                    ${legend}
                                </div>
                            </div>
                        </aside>
                    </div>

                    <section class="fact-grid">
                        ${facts}
                    </section>
                </main>
            </div>
        </div>
    `;
}

function buildScene(type, stage, config) {
    switch (type) {
        case "gpt":
            return buildGptScene(stage, config);
        case "hybrid":
            return buildHybridScene(stage, config);
        case "p20":
            return buildP20Scene(stage, config);
        case "fractal-v1":
            return buildFractalV1Scene(stage, config);
        case "fractal-v2":
            return buildFractalV2Scene(stage, config);
        default:
            throw new Error(`Unknown scene type: ${type}`);
    }
}

function buildGptScene(stage, config) {
    const root = new THREE.Group();
    stage.add(root);

    const tokenGroup = new THREE.Group();
    const cacheGroup = new THREE.Group();
    const layerGroup = new THREE.Group();
    const attentionGroup = new THREE.Group();
    const backwardGroup = new THREE.Group();
    const synthesisGroup = new THREE.Group();
    root.add(layerGroup, tokenGroup, cacheGroup, attentionGroup, backwardGroup, synthesisGroup);

    const tokenPositions = [];
    for (let i = 0; i < 12; i += 1) {
        const x = -8.25 + i * 1.5;
        tokenPositions.push(new THREE.Vector3(x, -4.8, 0));
        const token = box({
            width: 0.82,
            height: 0.42,
            depth: 0.82,
            color: config.accent,
            opacity: 0.86,
            emissive: 0.25,
        });
        token.position.set(x, -4.8, 0);
        token.userData.baseY = token.position.y;
        tokenGroup.add(token);

        const cache = box({
            width: 0.55,
            height: 0.2 + i * 0.16,
            depth: 0.55,
            color: config.accent,
            opacity: 0.22,
            emissive: 0.12,
        });
        cache.position.set(x, -4.2 + cache.geometry.parameters.height * 0.5, -3.7);
        cache.userData.baseHeight = cache.geometry.parameters.height;
        cacheGroup.add(cache);
    }

    const layers = [];
    for (let i = 0; i < 6; i += 1) {
        const slab = box({
            width: 18,
            height: 0.32,
            depth: 3.2,
            color: i % 2 === 0 ? config.accent2 : config.success,
            opacity: 0.16,
            emissive: 0.12,
        });
        slab.position.set(0, -2.5 + i * 1.35, 0);
        layers.push(slab);
        layerGroup.add(slab);
    }

    const arcs = [];
    const latest = tokenPositions[tokenPositions.length - 1];
    for (let i = 0; i < tokenPositions.length - 1; i += 1) {
        const target = tokenPositions[i];
        const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(latest.x, 1.6, 0.15),
            new THREE.Vector3((latest.x + target.x) * 0.5, 3.8, 1.5 + i * 0.08),
            new THREE.Vector3(target.x, 1.6, 0.15),
        ]);
        const line = curveLine(curve, config.accent2, 0.45);
        const pulse = pulseSphere(config.accent2, 0.13);
        arcs.push({ curve, line, pulse, speed: 0.15 + i * 0.012 });
        attentionGroup.add(line, pulse);
    }

    const fusion = box({
        width: 2.2,
        height: 1.0,
        depth: 2.2,
        color: config.success,
        opacity: 0.22,
        emissive: 0.15,
    });
    fusion.position.set(0, 4.7, 3.7);
    synthesisGroup.add(fusion);

    const lmHead = box({
        width: 1.6,
        height: 1.7,
        depth: 1.6,
        color: config.warning,
        opacity: 0.22,
        emissive: 0.18,
    });
    lmHead.position.set(0, 6.3, 5.2);
    synthesisGroup.add(lmHead);

    const synthesisLines = [];
    for (let i = 0; i < 8; i += 1) {
        const angle = (i / 8) * Math.PI * 2;
        const start = new THREE.Vector3(Math.cos(angle) * 4.5, 3.8, Math.sin(angle) * 1.25);
        const end = new THREE.Vector3(0, 4.7, 3.7);
        const line = straightLine(start, end, config.success, 0.5);
        const pulse = pulseSphere(config.success, 0.1);
        synthesisLines.push({ start, end, line, pulse, speed: 0.22 + i * 0.015 });
        synthesisGroup.add(line, pulse);
    }

    const backwardLines = [];
    for (let i = 0; i < 6; i += 1) {
        const start = new THREE.Vector3(0, 6.0 - i * 0.95, 3.0 - i * 0.36);
        const end = new THREE.Vector3(-6 + i * 2.1, -3.6, 0);
        const line = straightLine(start, end, config.danger, 0.36);
        const pulse = pulseSphere(config.danger, 0.12);
        backwardLines.push({ start, end, line, pulse, speed: 0.28 + i * 0.02 });
        backwardGroup.add(line, pulse);
    }

    return {
        setMode(mode) {
            attentionGroup.visible = mode !== "training";
            backwardGroup.visible = mode === "training";
            synthesisGroup.visible = mode === "synthesis";
            cacheGroup.visible = mode === "prefill" || mode === "decode";
        },
        update(t, mode) {
            tokenGroup.children.forEach((token, index) => {
                token.position.y = token.userData.baseY + Math.sin(t * 1.2 + index * 0.2) * 0.05;
                token.material.emissiveIntensity = mode === "decode" && index === tokenGroup.children.length - 1 ? 0.8 : 0.22;
            });
            cacheGroup.children.forEach((cache, index) => {
                const base = cache.userData.baseHeight;
                const scale = mode === "prefill"
                    ? 0.65 + 0.35 * (0.5 + 0.5 * Math.sin(t * 0.9 - index * 0.25))
                    : mode === "decode"
                        ? 0.9 + (index === cacheGroup.children.length - 1 ? 0.18 * Math.sin(t * 3.5) : 0)
                        : 1;
                cache.scale.y = scale;
                cache.position.y = -4.2 + (base * scale) * 0.5;
            });
            arcs.forEach((arc, index) => {
                const u = (t * arc.speed + index * 0.08) % 1;
                arc.pulse.position.copy(arc.curve.getPointAt(u));
                arc.line.material.opacity = mode === "decode" ? 0.75 : 0.45;
            });
            synthesisLines.forEach((bundle, index) => {
                const u = (t * bundle.speed + index * 0.1) % 1;
                bundle.pulse.position.copy(bundle.start.clone().lerp(bundle.end, u));
            });
            backwardLines.forEach((edge, index) => {
                const u = 1 - ((t * edge.speed + index * 0.08) % 1);
                edge.pulse.position.copy(edge.start.clone().lerp(edge.end, u));
            });
            root.rotation.y = Math.sin(t * 0.18) * 0.08;
            layerGroup.children.forEach((slab, index) => {
                slab.material.opacity = mode === "prefill" ? 0.18 + index * 0.01 : 0.14;
            });
        },
    };
}

function buildHybridScene(stage, config) {
    const root = new THREE.Group();
    stage.add(root);

    const tokenGroup = new THREE.Group();
    const statePath = new THREE.Group();
    const attentionLayers = new THREE.Group();
    const fusionGroup = new THREE.Group();
    const backwardGroup = new THREE.Group();
    root.add(tokenGroup, statePath, attentionLayers, fusionGroup, backwardGroup);

    const pathPoints = [];
    for (let i = 0; i < 12; i += 1) {
        const x = -8 + i * 1.45;
        pathPoints.push(new THREE.Vector3(x, -3.6 + Math.sin(i * 0.35) * 0.45, 1.4));
        const token = box({
            width: 0.75,
            height: 0.36,
            depth: 0.75,
            color: config.accent,
            opacity: 0.84,
            emissive: 0.22,
        });
        token.position.set(x, -4.8, -0.4);
        tokenGroup.add(token);
    }

    const stateCurve = new THREE.CatmullRomCurve3(pathPoints);
    const stateTube = new THREE.Mesh(
        new THREE.TubeGeometry(stateCurve, 128, 0.19, 12, false),
        new THREE.MeshStandardMaterial({
            color: hex(config.accent),
            transparent: true,
            opacity: 0.72,
            emissive: hex(config.accent),
            emissiveIntensity: 0.28,
            roughness: 0.3,
            metalness: 0.1,
        }),
    );
    statePath.add(stateTube);

    const statePulse = pulseSphere(config.accent, 0.22);
    statePath.add(statePulse);

    const stateNodes = [];
    for (let i = 0; i < 8; i += 1) {
        const u = i / 7;
        const node = box({
            width: 0.55,
            height: 0.55,
            depth: 0.55,
            color: config.success,
            opacity: 0.2,
            emissive: 0.14,
        });
        node.position.copy(stateCurve.getPointAt(u));
        stateNodes.push(node);
        statePath.add(node);
    }

    const alternating = [];
    for (let i = 0; i < 6; i += 1) {
        const isAttention = i % 2 === 0;
        const slab = box({
            width: 15.5,
            height: 0.36,
            depth: 2.8,
            color: isAttention ? config.accent2 : config.accent,
            opacity: isAttention ? 0.18 : 0.14,
            emissive: 0.12,
        });
        slab.position.set(0, -1.8 + i * 1.18, 0);
        alternating.push({ slab, isAttention });
        if (isAttention) {
            attentionLayers.add(slab);
            const bridge = straightLine(
                new THREE.Vector3(-5.5, slab.position.y + 0.15, -0.3),
                new THREE.Vector3(5.5, slab.position.y + 0.15, -0.3),
                config.accent2,
                0.42,
            );
            const bridgePulse = pulseSphere(config.accent2, 0.12);
            attentionLayers.add(bridge, bridgePulse);
            alternating[alternating.length - 1].bridge = bridge;
            alternating[alternating.length - 1].bridgePulse = bridgePulse;
        } else {
            statePath.add(slab);
        }
    }

    const fusion = box({
        width: 2.0,
        height: 1.0,
        depth: 2.0,
        color: config.success,
        opacity: 0.2,
        emissive: 0.16,
    });
    fusion.position.set(0, 5.4, 3.4);
    fusionGroup.add(fusion);

    const lmHead = box({
        width: 1.4,
        height: 1.5,
        depth: 1.4,
        color: config.warning,
        opacity: 0.24,
        emissive: 0.18,
    });
    lmHead.position.set(0, 6.8, 4.7);
    fusionGroup.add(lmHead);

    const fusionEdges = [];
    for (let i = 0; i < 5; i += 1) {
        const start = new THREE.Vector3(-4 + i * 2, 4.0, -0.4 + (i % 2 ? 0.7 : -0.2));
        const end = new THREE.Vector3(0, 5.4, 3.4);
        const line = straightLine(start, end, i % 2 ? config.accent2 : config.success, 0.45);
        const pulse = pulseSphere(i % 2 ? config.accent2 : config.success, 0.1);
        fusionGroup.add(line, pulse);
        fusionEdges.push({ start, end, pulse, speed: 0.23 + i * 0.02 });
    }

    const backwardEdges = [];
    for (let i = 0; i < 6; i += 1) {
        const start = new THREE.Vector3(0, 6.2 - i * 0.85, 2.8 - i * 0.25);
        const end = new THREE.Vector3(-6 + i * 2.2, -3.8, -0.2);
        const line = straightLine(start, end, config.danger, 0.32);
        const pulse = pulseSphere(config.danger, 0.12);
        backwardGroup.add(line, pulse);
        backwardEdges.push({ start, end, pulse, speed: 0.28 + i * 0.025 });
    }

    return {
        setMode(mode) {
            backwardGroup.visible = mode === "training";
            attentionLayers.visible = true;
            fusionGroup.visible = mode === "synthesis";
        },
        update(t, mode) {
            statePulse.position.copy(stateCurve.getPointAt((t * 0.18) % 1));
            stateTube.material.emissiveIntensity = mode === "decode" ? 0.42 : 0.28;
            alternating.forEach((layer, index) => {
                layer.slab.material.opacity =
                    layer.isAttention
                        ? mode === "decode"
                            ? index % 4 === 0
                                ? 0.22
                                : 0.08
                            : 0.18
                        : mode === "prefill"
                            ? 0.2
                            : 0.14;
                if (layer.bridgePulse) {
                    const u = (t * (0.18 + index * 0.02)) % 1;
                    layer.bridgePulse.position.copy(
                        new THREE.Vector3(-5.5 + 11 * u, layer.slab.position.y + 0.15, -0.3),
                    );
                    layer.bridge.material.opacity = mode === "decode" ? 0.22 : 0.42;
                }
            });
            stateNodes.forEach((node, index) => {
                node.material.opacity = 0.14 + 0.1 * (0.5 + 0.5 * Math.sin(t * 2 + index));
            });
            fusionEdges.forEach((edge, index) => {
                const u = (t * edge.speed + index * 0.1) % 1;
                edge.pulse.position.copy(edge.start.clone().lerp(edge.end, u));
            });
            backwardEdges.forEach((edge, index) => {
                const u = 1 - ((t * edge.speed + index * 0.08) % 1);
                edge.pulse.position.copy(edge.start.clone().lerp(edge.end, u));
            });
            root.rotation.y = Math.sin(t * 0.16) * 0.08;
        },
    };
}

function buildP20Scene(stage, config) {
    const root = new THREE.Group();
    stage.add(root);

    const tokenGroup = new THREE.Group();
    const preludeGroup = new THREE.Group();
    const projectionGroup = new THREE.Group();
    const stateGroup = new THREE.Group();
    const loopGroup = new THREE.Group();
    const outputGroup = new THREE.Group();
    const backwardGroup = new THREE.Group();
    const boundaryGroup = new THREE.Group();
    root.add(tokenGroup, preludeGroup, projectionGroup, stateGroup, loopGroup, outputGroup, backwardGroup, boundaryGroup);

    const tokenPositions = [];
    for (let i = 0; i < 12; i += 1) {
        const x = -8.4 + i * 1.15;
        const z = -3.8 + Math.sin(i * 0.45) * 0.22;
        tokenPositions.push(new THREE.Vector3(x, -4.8, z));
        const token = box({
            width: 0.62,
            height: 0.36,
            depth: 0.62,
            color: config.accent,
            opacity: 0.82,
            emissive: 0.24,
        });
        token.position.copy(tokenPositions[i]);
        token.userData.baseY = token.position.y;
        tokenGroup.add(token);
    }

    const preludeSlabs = [];
    for (let i = 0; i < 3; i += 1) {
        const slab = box({
            width: 12.6,
            height: 0.28,
            depth: 2.2,
            color: i % 2 === 0 ? config.accent2 : config.success,
            opacity: 0.13,
            emissive: 0.12,
        });
        slab.position.set(-3.1, -3.2 + i * 0.75, -0.3);
        preludeSlabs.push(slab);
        preludeGroup.add(slab);
    }

    const packedProj = box({
        width: 1.5,
        height: 2.6,
        depth: 1.0,
        color: config.warning,
        opacity: 0.24,
        emissive: 0.22,
    });
    packedProj.position.set(-3.6, -0.6, 0.1);
    projectionGroup.add(packedProj);

    const inputCurve = new THREE.CatmullRomCurve3([
        tokenPositions[0].clone(),
        new THREE.Vector3(-6.2, -3.8, -2.4),
        new THREE.Vector3(-4.8, -2.0, -0.8),
        packedProj.position.clone(),
    ]);
    const inputLine = curveLine(inputCurve, config.accent, 0.55);
    const inputPulse = pulseSphere(config.accent, 0.14);
    tokenGroup.add(inputLine, inputPulse);

    const controls = [];
    const controlSpecs = [
        { name: "update", color: config.warning, position: new THREE.Vector3(-1.8, 1.7, -1.9) },
        { name: "angle", color: config.accent2, position: new THREE.Vector3(-1.1, 0.4, -2.4) },
        { name: "candidate", color: config.success, position: new THREE.Vector3(-1.4, -1.0, -2.0) },
        { name: "output", color: config.danger, position: new THREE.Vector3(-2.1, -2.2, -1.2) },
    ];
    const stateCenter = new THREE.Vector3(0.3, -0.2, 0.3);
    controlSpecs.forEach((spec, index) => {
        const node = box({
            width: 0.72,
            height: 0.72,
            depth: 0.72,
            color: spec.color,
            opacity: 0.26,
            emissive: 0.2,
        });
        node.position.copy(spec.position);
        const fromPacked = straightLine(packedProj.position, spec.position, spec.color, 0.36);
        const toState = straightLine(spec.position, stateCenter, spec.color, 0.38);
        const pulse = pulseSphere(spec.color, 0.09);
        projectionGroup.add(node, fromPacked, toState, pulse);
        controls.push({ spec, node, pulse, speed: 0.18 + index * 0.028 });
    });

    const stateOrb = new THREE.Mesh(
        new THREE.IcosahedronGeometry(1.1, 1),
        new THREE.MeshStandardMaterial({
            color: hex(config.success),
            transparent: true,
            opacity: 0.32,
            emissive: hex(config.success),
            emissiveIntensity: 0.34,
            roughness: 0.22,
            metalness: 0.14,
        }),
    );
    stateOrb.position.copy(stateCenter);
    stateGroup.add(stateOrb);

    const rings = [];
    for (let i = 0; i < 4; i += 1) {
        const ring = new THREE.Mesh(
            new THREE.TorusGeometry(1.5 + i * 0.22, 0.045, 10, 90),
            new THREE.MeshStandardMaterial({
                color: i % 2 === 0 ? hex(config.accent2) : hex(config.warning),
                transparent: true,
                opacity: 0.42,
                emissive: i % 2 === 0 ? hex(config.accent2) : hex(config.warning),
                emissiveIntensity: 0.2,
            }),
        );
        ring.position.copy(stateCenter);
        ring.rotation.set(i * 0.55, i * 0.32, i * 0.7);
        stateGroup.add(ring);
        rings.push(ring);
    }

    const recurrentCarry = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-1.2, -0.2, 0.3),
        new THREE.Vector3(-0.9, 1.8, 1.1),
        new THREE.Vector3(1.4, 1.6, 0.9),
        new THREE.Vector3(1.7, -0.2, 0.3),
        new THREE.Vector3(0.3, -0.2, 0.3),
    ]);
    const carryLine = curveLine(recurrentCarry, config.success, 0.5);
    const carryPulse = pulseSphere(config.success, 0.13);
    stateGroup.add(carryLine, carryPulse);

    const loopChamber = box({
        width: 4.9,
        height: 3.0,
        depth: 3.3,
        color: config.accent2,
        opacity: 0.08,
        emissive: 0.12,
    });
    loopChamber.position.set(5.2, 0.1, 0.2);
    loopGroup.add(loopChamber);

    const middleBlocks = [];
    for (let i = 0; i < 4; i += 1) {
        const slab = box({
            width: 4.4,
            height: 0.26,
            depth: 2.5,
            color: i % 2 === 0 ? config.accent2 : config.success,
            opacity: 0.18,
            emissive: 0.14,
        });
        slab.position.set(5.2, -1.05 + i * 0.7, 0.2);
        loopGroup.add(slab);
        middleBlocks.push(slab);
    }

    const controlCurve = new THREE.CatmullRomCurve3([
        stateCenter.clone(),
        new THREE.Vector3(2.0, 0.8, 1.6),
        new THREE.Vector3(3.7, 1.2, 1.2),
        loopChamber.position.clone(),
    ]);
    const controlLine = curveLine(controlCurve, config.success, 0.58);
    const controlPulse = pulseSphere(config.success, 0.16);
    loopGroup.add(controlLine, controlPulse);

    const loopCurves = [];
    for (let i = 0; i < 3; i += 1) {
        const y = -0.75 + i * 0.7;
        const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(3.1, y, -1.0),
            new THREE.Vector3(5.2, y + 1.0, -1.8),
            new THREE.Vector3(7.3, y, -1.0),
            new THREE.Vector3(5.2, y - 0.9, 1.7),
            new THREE.Vector3(3.1, y, -1.0),
        ], true);
        const line = curveLine(curve, config.accent2, 0.34);
        const pulse = pulseSphere(config.accent2, 0.1);
        loopGroup.add(line, pulse);
        loopCurves.push({ curve, pulse, speed: 0.13 + i * 0.035 });
    }

    const coda = box({
        width: 2.4,
        height: 1.2,
        depth: 2.0,
        color: config.accent,
        opacity: 0.18,
        emissive: 0.14,
    });
    coda.position.set(8.7, 1.2, 0.2);
    outputGroup.add(coda);

    const lmHead = box({
        width: 1.4,
        height: 1.7,
        depth: 1.4,
        color: config.warning,
        opacity: 0.25,
        emissive: 0.2,
    });
    lmHead.position.set(10.2, 2.7, 1.2);
    outputGroup.add(lmHead);
    const outputLine = straightLine(loopChamber.position, coda.position, config.accent, 0.45);
    const lmLine = straightLine(coda.position, lmHead.position, config.warning, 0.42);
    const outputPulse = pulseSphere(config.accent, 0.11);
    outputGroup.add(outputLine, lmLine, outputPulse);

    const backwardEdges = [];
    for (let i = 0; i < 8; i += 1) {
        const start = new THREE.Vector3(9.5 - i * 1.1, 2.7 - i * 0.32, 2.0 - i * 0.26);
        const end = new THREE.Vector3(-4.2 + i * 0.7, -3.5, -1.3 + i * 0.12);
        const line = straightLine(start, end, config.danger, 0.24);
        const pulse = pulseSphere(config.danger, 0.1);
        backwardGroup.add(line, pulse);
        backwardEdges.push({ start, end, pulse, speed: 0.22 + i * 0.018 });
    }

    const boundaryBars = [
        { label: "12L", loss: 4.3304, x: -1.8, color: config.accent },
        { label: "14L", loss: 4.3022, x: 0.0, color: config.warning },
        { label: "P20", loss: 4.2983, x: 1.8, color: config.success },
        { label: "16L", loss: 4.2758, x: 3.6, color: config.danger },
    ];
    boundaryBars.forEach((entry) => {
        const height = 1.0 + (4.36 - entry.loss) * 10;
        const bar = box({
            width: 0.7,
            height,
            depth: 0.7,
            color: entry.color,
            opacity: 0.34,
            emissive: 0.18,
        });
        bar.position.set(entry.x, -4.9 + height * 0.5, 4.2);
        bar.userData.baseHeight = height;
        boundaryGroup.add(bar);
    });
    boundaryGroup.add(straightLine(new THREE.Vector3(-2.4, -4.9, 4.2), new THREE.Vector3(4.2, -4.9, 4.2), config.warning, 0.28));

    return {
        setMode(mode) {
            projectionGroup.visible = mode !== "loop" && mode !== "boundary";
            preludeGroup.visible = mode !== "boundary";
            tokenGroup.visible = mode !== "boundary";
            loopGroup.visible = mode !== "scan" && mode !== "boundary";
            outputGroup.visible = mode !== "scan" && mode !== "boundary";
            backwardGroup.visible = mode === "training";
            boundaryGroup.visible = mode === "boundary";
        },
        update(t, mode) {
            tokenGroup.children.forEach((child, index) => {
                if (!child.geometry || !child.userData) {
                    return;
                }
                if (child.userData.baseY !== undefined) {
                    child.position.y = child.userData.baseY + Math.sin(t * 1.4 + index * 0.4) * 0.05;
                    child.material.emissiveIntensity = mode === "scan" ? 0.34 : 0.22;
                }
            });
            inputPulse.position.copy(inputCurve.getPointAt((t * 0.2) % 1));
            preludeSlabs.forEach((slab, index) => {
                slab.material.opacity = 0.11 + 0.06 * (0.5 + 0.5 * Math.sin(t * 1.6 + index));
            });
            controls.forEach((control, index) => {
                const phase = (t * control.speed + index * 0.15) % 1;
                const start = phase < 0.45
                    ? packedProj.position
                    : control.spec.position;
                const end = phase < 0.45
                    ? control.spec.position
                    : stateCenter;
                const u = phase < 0.45 ? phase / 0.45 : (phase - 0.45) / 0.55;
                control.pulse.position.copy(start.clone().lerp(end, u));
                control.node.material.opacity =
                    mode === "anatomy" || mode === "scan"
                        ? 0.26 + 0.12 * Math.sin(t * 2.0 + index)
                        : 0.16;
            });
            stateOrb.rotation.x += 0.004;
            stateOrb.rotation.y += 0.006;
            stateOrb.material.emissiveIntensity = mode === "scan" ? 0.55 : 0.34;
            rings.forEach((ring, index) => {
                ring.rotation.x += 0.003 + index * 0.0007;
                ring.rotation.z += 0.005 + index * 0.001;
                ring.material.opacity = mode === "scan" ? 0.55 : 0.4;
            });
            carryPulse.position.copy(recurrentCarry.getPointAt((t * 0.18) % 1));
            controlPulse.position.copy(controlCurve.getPointAt((t * 0.16) % 1));
            middleBlocks.forEach((slab, index) => {
                slab.material.opacity =
                    mode === "loop"
                        ? 0.22 + 0.08 * (0.5 + 0.5 * Math.sin(t * 1.8 + index))
                        : 0.16;
            });
            loopCurves.forEach((entry, index) => {
                const u = (t * entry.speed + index * 0.14) % 1;
                entry.pulse.position.copy(entry.curve.getPointAt(u));
            });
            const outputU = (t * 0.18) % 1;
            outputPulse.position.copy(loopChamber.position.clone().lerp(coda.position, outputU));
            backwardEdges.forEach((edge, index) => {
                const u = 1 - ((t * edge.speed + index * 0.08) % 1);
                edge.pulse.position.copy(edge.start.clone().lerp(edge.end, u));
            });
            boundaryGroup.children.forEach((child, index) => {
                if (child.material && child.userData.baseHeight) {
                    child.scale.y = 1 + 0.04 * Math.sin(t * 1.7 + index);
                    child.material.opacity = 0.28 + 0.08 * (0.5 + 0.5 * Math.sin(t * 1.2 + index));
                }
            });
            root.rotation.y = Math.sin(t * 0.14) * 0.08;
        },
    };
}

function buildFractalV1Scene(stage, config) {
    const root = new THREE.Group();
    stage.add(root);

    const tokenFlow = new THREE.Group();
    const stateGroup = new THREE.Group();
    const routerGroup = new THREE.Group();
    const graphGroup = new THREE.Group();
    const failureGroup = new THREE.Group();
    root.add(tokenFlow, stateGroup, routerGroup, graphGroup, failureGroup);

    const entryCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-9, -2.8, -1),
        new THREE.Vector3(-5.5, -1.2, -0.2),
        new THREE.Vector3(-2.2, -0.5, 0.6),
        new THREE.Vector3(0, 0, 0),
    ]);
    const entryTube = new THREE.Mesh(
        new THREE.TubeGeometry(entryCurve, 100, 0.16, 10, false),
        new THREE.MeshStandardMaterial({
            color: hex(config.accent),
            transparent: true,
            opacity: 0.74,
            emissive: hex(config.accent),
            emissiveIntensity: 0.25,
        }),
    );
    tokenFlow.add(entryTube);
    const entryPulse = pulseSphere(config.accent, 0.18);
    tokenFlow.add(entryPulse);

    const stateOrb = new THREE.Mesh(
        new THREE.IcosahedronGeometry(1.45, 1),
        new THREE.MeshStandardMaterial({
            color: hex(config.success),
            transparent: true,
            opacity: 0.32,
            emissive: hex(config.success),
            emissiveIntensity: 0.35,
            roughness: 0.25,
            metalness: 0.18,
        }),
    );
    stateGroup.add(stateOrb);

    const rings = [];
    for (let i = 0; i < 3; i += 1) {
        const ring = new THREE.Mesh(
            new THREE.TorusGeometry(2.0 + i * 0.55, 0.06, 12, 120),
            new THREE.MeshStandardMaterial({
                color: i === 1 ? hex(config.warning) : hex(config.accent2),
                transparent: true,
                opacity: 0.48,
                emissive: i === 1 ? hex(config.warning) : hex(config.accent2),
                emissiveIntensity: 0.18,
            }),
        );
        ring.rotation.set(i * 0.7, i * 0.35, i * 0.48);
        stateGroup.add(ring);
        rings.push(ring);
    }

    const projections = [];
    const projectionPositions = [
        new THREE.Vector3(-3.8, -1.9, 1.9),
        new THREE.Vector3(-2.5, 2.3, -2.1),
        new THREE.Vector3(2.9, -1.5, -2.3),
    ];
    projectionPositions.forEach((position, index) => {
        const node = box({
            width: 0.7,
            height: 1.7,
            depth: 0.7,
            color: index === 1 ? config.warning : config.accent2,
            opacity: 0.22,
            emissive: 0.18,
        });
        node.position.copy(position);
        const line = straightLine(position, new THREE.Vector3(0, 0, 0), config.success, 0.38);
        const pulse = pulseSphere(config.success, 0.09);
        projections.push({ position, line, pulse, speed: 0.24 + index * 0.03 });
        stateGroup.add(node, line, pulse);
    });

    const router = diamond(config.warning, 1.15);
    router.position.set(3.8, 2.2, 0.5);
    routerGroup.add(router);
    const routerLine = straightLine(new THREE.Vector3(0, 0, 0), router.position, config.warning, 0.4);
    routerGroup.add(routerLine);

    const outputHead = box({
        width: 1.6,
        height: 1.9,
        depth: 1.6,
        color: config.accent,
        opacity: 0.18,
        emissive: 0.16,
    });
    outputHead.position.set(6.2, 0.4, 0);
    routerGroup.add(outputHead);
    routerGroup.add(straightLine(router.position, outputHead.position, config.accent, 0.4));

    const graphPlanes = [];
    for (let i = 0; i < 9; i += 1) {
        const plane = box({
            width: 7.2,
            height: 0.12,
            depth: 4.6,
            color: config.danger,
            opacity: 0.06,
            emissive: 0.08,
        });
        plane.position.set(0, -4.5 + i * 0.78, -4.6 - i * 0.42);
        graphGroup.add(plane);
        graphPlanes.push(plane);
    }

    const graphEdges = [];
    for (let row = 0; row < 5; row += 1) {
        for (let col = 0; col < 4; col += 1) {
            const start = new THREE.Vector3(-4 + row * 2, -3.8 + col * 1.2, -3.8 - row * 0.55);
            const end = new THREE.Vector3(start.x + 1.7, start.y + 0.85, start.z - 1.05);
            const line = straightLine(start, end, config.danger, 0.22);
            const pulse = pulseSphere(config.danger, 0.08);
            graphGroup.add(line, pulse);
            graphEdges.push({ start, end, pulse, speed: 0.18 + row * 0.015 });
        }
    }

    const failureBurst = new THREE.Group();
    const failureShell = new THREE.Mesh(
        new THREE.TorusKnotGeometry(2.7, 0.18, 180, 18, 3, 5),
        new THREE.MeshStandardMaterial({
            color: hex(config.danger),
            transparent: true,
            opacity: 0.18,
            emissive: hex(config.danger),
            emissiveIntensity: 0.42,
        }),
    );
    failureShell.position.set(0, 0, 0);
    failureBurst.add(failureShell);
    for (let i = 0; i < 14; i += 1) {
        const angle = (i / 14) * Math.PI * 2;
        const start = new THREE.Vector3(Math.cos(angle) * 1.2, Math.sin(angle * 2) * 0.8, Math.sin(angle) * 1.2);
        const end = start.clone().multiplyScalar(3.4);
        failureBurst.add(straightLine(start, end, config.danger, 0.35));
    }
    failureGroup.add(failureBurst);

    return {
        setMode(mode) {
            tokenFlow.visible = true;
            stateGroup.visible = true;
            routerGroup.visible = true;
            graphGroup.visible = mode === "backward" || mode === "failure";
            failureGroup.visible = mode === "failure";
        },
        update(t, mode) {
            entryPulse.position.copy(entryCurve.getPointAt((t * 0.2) % 1));
            stateOrb.rotation.x += 0.002;
            stateOrb.rotation.y += 0.003;
            stateOrb.material.emissiveIntensity = mode === "forward" ? 0.46 : 0.32;
            rings.forEach((ring, index) => {
                ring.rotation.x += 0.002 + index * 0.0008;
                ring.rotation.z += 0.003 + index * 0.0009;
                ring.material.opacity = mode === "structure" ? 0.42 : 0.55;
            });
            projections.forEach((projection, index) => {
                const u = (t * projection.speed + index * 0.15) % 1;
                projection.pulse.position.copy(projection.position.clone().lerp(new THREE.Vector3(0, 0, 0), u));
            });
            router.rotation.y += 0.01;
            graphPlanes.forEach((plane, index) => {
                plane.material.opacity =
                    mode === "backward"
                        ? 0.07 + 0.05 * (0.5 + 0.5 * Math.sin(t * 2.2 - index * 0.3))
                        : mode === "failure"
                            ? 0.1 + 0.06 * (0.5 + 0.5 * Math.sin(t * 2.8 - index * 0.25))
                            : 0.04;
            });
            graphEdges.forEach((edge, index) => {
                const u = 1 - ((t * edge.speed + index * 0.06) % 1);
                edge.pulse.position.copy(edge.start.clone().lerp(edge.end, u));
            });
            failureBurst.rotation.x += 0.004;
            failureBurst.rotation.y += 0.006;
            failureShell.material.opacity = mode === "failure" ? 0.28 + 0.06 * Math.sin(t * 2.2) : 0.1;
            root.rotation.y = Math.sin(t * 0.18) * 0.1;
        },
    };
}

function buildFractalV2Scene(stage, config) {
    const root = new THREE.Group();
    stage.add(root);

    const leafGroup = new THREE.Group();
    const rootGroup = new THREE.Group();
    const treeGroup = new THREE.Group();
    const routeGroup = new THREE.Group();
    const fusionGroup = new THREE.Group();
    const teacherGroup = new THREE.Group();
    root.add(leafGroup, rootGroup, treeGroup, routeGroup, fusionGroup, teacherGroup);

    const leafCenters = [];
    const leaves = [];
    for (let i = 0; i < 8; i += 1) {
        const x = -8.4 + i * 2.4;
        leafCenters.push(new THREE.Vector3(x, -4.4, 0));
        const leaf = box({
            width: 1.6,
            height: 0.7,
            depth: 1.25,
            color: i % 2 === 0 ? config.accent : config.accent2,
            opacity: 0.24,
            emissive: 0.16,
        });
        leaf.position.set(x, -4.4, 0);
        leaf.userData.baseY = leaf.position.y;
        leafGroup.add(leaf);
        leaves.push(leaf);
    }

    const trunks = [];
    for (let r = 0; r < 4; r += 1) {
        const lane = new THREE.Group();
        const z = -3.4 + r * 2.2;
        for (let i = 0; i < leafCenters.length; i += 1) {
            const cell = box({
                width: 0.55,
                height: 0.55,
                depth: 0.55,
                color: r % 2 === 0 ? config.success : config.accent,
                opacity: 0.22,
                emissive: 0.16,
            });
            cell.position.set(leafCenters[i].x, -6.2, z);
            lane.add(cell);
        }
        const line = straightLine(
            new THREE.Vector3(leafCenters[0].x, -6.2, z),
            new THREE.Vector3(leafCenters[leafCenters.length - 1].x, -6.2, z),
            r % 2 === 0 ? config.success : config.accent,
            0.38,
        );
        lane.add(line);
        rootGroup.add(lane);
        trunks.push(lane);
    }

    const treeNodes = [];
    let currentLevel = leafCenters.map((center) => center.clone());
    for (let level = 0; level < 3; level += 1) {
        const nextLevel = [];
        for (let i = 0; i < currentLevel.length; i += 2) {
            const left = currentLevel[i];
            const right = currentLevel[i + 1];
            const mid = new THREE.Vector3((left.x + right.x) * 0.5, -1.6 + level * 2.2, 0.6 + level * 0.4);
            nextLevel.push(mid);
            const node = box({
                width: 1.1 - level * 0.12,
                height: 0.5,
                depth: 1.1 - level * 0.12,
                color: config.accent2,
                opacity: 0.18,
                emissive: 0.16,
            });
            node.position.copy(mid);
            treeGroup.add(node);
            treeGroup.add(straightLine(left, mid, config.accent2, 0.36));
            treeGroup.add(straightLine(right, mid, config.accent2, 0.36));
            treeNodes.push({ node, mid, level });
        }
        currentLevel = nextLevel;
    }

    const routingHeads = [];
    for (let i = 0; i < 4; i += 1) {
        const color = i % 2 === 0 ? config.accent2 : config.warning;
        const sphere = pulseSphere(color, 0.16);
        routeGroup.add(sphere);
        const coarseTarget = treeNodes[i % treeNodes.length].mid.clone();
        const leafTarget = leafCenters[(i * 2 + 1) % leafCenters.length].clone();
        routingHeads.push({
            sphere,
            root: currentLevel[0].clone().add(new THREE.Vector3(0.2 * (i - 1.5), 2.2, 0.8)),
            coarseTarget,
            leafTarget,
            speed: 0.18 + i * 0.03,
        });
    }

    const memoryBank = new THREE.Group();
    const slots = [];
    for (let i = 0; i < 6; i += 1) {
        const slot = box({
            width: 0.65,
            height: 1.0 + (i % 3) * 0.22,
            depth: 0.65,
            color: config.success,
            opacity: 0.16,
            emissive: 0.14,
        });
        slot.position.set(9.2, -3.4 + i * 1.18, 2.8);
        memoryBank.add(slot);
        slots.push(slot);
    }
    fusionGroup.add(memoryBank);

    const fusion = box({
        width: 2.2,
        height: 1.0,
        depth: 2.2,
        color: config.success,
        opacity: 0.2,
        emissive: 0.16,
    });
    fusion.position.set(0, 6.0, 4.1);
    fusionGroup.add(fusion);

    const lmHead = box({
        width: 1.6,
        height: 1.8,
        depth: 1.6,
        color: config.warning,
        opacity: 0.22,
        emissive: 0.18,
    });
    lmHead.position.set(0, 7.4, 5.5);
    fusionGroup.add(lmHead);

    const fusionBeams = [];
    for (let i = 0; i < 4; i += 1) {
        const start = new THREE.Vector3(-5 + i * 3.3, -6.2, -1.8 + i * 0.9);
        const end = new THREE.Vector3(0, 6.0, 4.1);
        const line = straightLine(start, end, config.success, 0.32);
        const pulse = pulseSphere(config.success, 0.1);
        fusionGroup.add(line, pulse);
        fusionBeams.push({ start, end, pulse, speed: 0.14 + i * 0.03 });
    }

    const leafPointer = straightLine(leafCenters[5], new THREE.Vector3(5.2, 0.8, 2.6), config.warning, 0.44);
    const pointerPulse = pulseSphere(config.warning, 0.12);
    fusionGroup.add(leafPointer, pointerPulse);

    const teacherRing = new THREE.Mesh(
        new THREE.TorusGeometry(3.4, 0.05, 10, 100),
        new THREE.MeshStandardMaterial({
            color: hex(config.warning),
            transparent: true,
            opacity: 0.2,
            emissive: hex(config.warning),
            emissiveIntensity: 0.22,
        }),
    );
    teacherRing.rotation.x = Math.PI / 2;
    teacherRing.position.set(0, 3.0, -5.4);
    teacherGroup.add(teacherRing);
    const teacherLines = [];
    for (let i = 0; i < 3; i += 1) {
        const target = treeNodes[i * 2].mid.clone();
        const line = straightLine(teacherRing.position, target, config.warning, 0.22);
        const pulse = pulseSphere(config.warning, 0.1);
        teacherGroup.add(line, pulse);
        teacherLines.push({ start: teacherRing.position.clone(), end: target, pulse, speed: 0.12 + i * 0.03 });
    }

    return {
        setMode(mode) {
            teacherGroup.visible = mode === "training";
            pointerPulse.visible = mode === "decode" || mode === "synthesis";
            leafPointer.visible = mode === "decode" || mode === "synthesis";
        },
        update(t, mode) {
            leaves.forEach((leaf, index) => {
                const offset =
                    mode === "prefill"
                        ? 0.12 * Math.max(0, Math.sin(t * 1.6 - index * 0.32))
                        : mode === "decode" && index === leaves.length - 1
                            ? 0.16 * Math.sin(t * 4)
                            : 0;
                leaf.position.y = leaf.userData.baseY + offset;
                leaf.material.opacity = mode === "decode" && index === leaves.length - 1 ? 0.36 : 0.24;
            });
            trunks.forEach((lane, laneIndex) => {
                lane.children.forEach((child, childIndex) => {
                    if (child.material) {
                        child.material.opacity = 0.18 + 0.08 * (0.5 + 0.5 * Math.sin(t * 1.8 + laneIndex + childIndex * 0.2));
                    }
                });
            });
            treeNodes.forEach((entry, index) => {
                entry.node.material.opacity =
                    mode === "prefill"
                        ? 0.16 + 0.08 * Math.max(0, Math.sin(t * 1.5 - index * 0.2))
                        : mode === "decode"
                            ? index >= treeNodes.length - 2
                                ? 0.28
                                : 0.12
                            : 0.18;
            });
            routingHeads.forEach((head, index) => {
                const phase = (t * head.speed + index * 0.12) % 1;
                let position;
                if (mode === "prefill") {
                    position = head.root.clone().lerp(head.coarseTarget, phase);
                } else if (mode === "decode") {
                    position = phase < 0.45
                        ? head.root.clone().lerp(head.coarseTarget, phase / 0.45)
                        : head.coarseTarget.clone().lerp(head.leafTarget, (phase - 0.45) / 0.55);
                } else if (mode === "synthesis") {
                    position = phase < 0.5
                        ? head.root.clone().lerp(head.coarseTarget, phase / 0.5)
                        : head.coarseTarget.clone().lerp(new THREE.Vector3(0, 6.0, 4.1), (phase - 0.5) / 0.5);
                } else {
                    position = phase < 0.4
                        ? head.root.clone().lerp(head.coarseTarget, phase / 0.4)
                        : head.coarseTarget.clone().lerp(head.leafTarget, (phase - 0.4) / 0.6);
                }
                head.sphere.position.copy(position);
            });
            slots.forEach((slot, index) => {
                slot.material.opacity = 0.14 + 0.08 * (0.5 + 0.5 * Math.sin(t * 2.1 + index));
            });
            fusionBeams.forEach((beam, index) => {
                const u = (t * beam.speed + index * 0.1) % 1;
                beam.pulse.position.copy(beam.start.clone().lerp(beam.end, u));
            });
            const pointerU = (t * 0.22) % 1;
            pointerPulse.position.copy(leafCenters[5].clone().lerp(new THREE.Vector3(5.2, 0.8, 2.6), pointerU));
            teacherRing.rotation.z += 0.004;
            teacherLines.forEach((beam, index) => {
                const u = (t * beam.speed + index * 0.15) % 1;
                beam.pulse.position.copy(beam.start.clone().lerp(beam.end, u));
            });
            root.rotation.y = Math.sin(t * 0.16) * 0.08;
        },
    };
}

function makeSoftControls(canvas, stage, camera) {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let rotX = -0.24;
    let rotY = 0.48;
    let velX = 0;
    let velY = 0;

    canvas.addEventListener("pointerdown", (event) => {
        dragging = true;
        lastX = event.clientX;
        lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
        if (!dragging) {
            return;
        }
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        lastX = event.clientX;
        lastY = event.clientY;
        rotY += dx * 0.0045;
        rotX += dy * 0.0038;
        rotX = clamp(rotX, -1.15, 1.15);
        velY = dx * 0.0007;
        velX = dy * 0.00055;
    });

    function stopDrag(event) {
        dragging = false;
        if (event) {
            canvas.releasePointerCapture(event.pointerId);
        }
    }

    canvas.addEventListener("pointerup", stopDrag);
    canvas.addEventListener("pointerleave", stopDrag);
    canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        camera.position.z = clamp(camera.position.z + event.deltaY * 0.012, 10, 34);
    }, { passive: false });

    return {
        update() {
            if (!dragging) {
                rotY += 0.0012 + velY;
                velY *= 0.96;
                velX *= 0.96;
                rotX += velX;
            }
            stage.rotation.x = rotX;
            stage.rotation.y = rotY;
        },
    };
}

function makeStarField() {
    const geometry = new THREE.BufferGeometry();
    const points = [];
    for (let i = 0; i < 900; i += 1) {
        points.push(
            (Math.random() - 0.5) * 70,
            (Math.random() - 0.5) * 46,
            (Math.random() - 0.5) * 70,
        );
    }
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
    const material = new THREE.PointsMaterial({
        color: 0xa8c9ff,
        size: 0.06,
        transparent: true,
        opacity: 0.55,
    });
    return new THREE.Points(geometry, material);
}

function box({ width, height, depth, color, opacity, emissive }) {
    return new THREE.Mesh(
        new THREE.BoxGeometry(width, height, depth),
        new THREE.MeshStandardMaterial({
            color: hex(color),
            transparent: true,
            opacity,
            emissive: hex(color),
            emissiveIntensity: emissive || 0.15,
            roughness: 0.28,
            metalness: 0.12,
        }),
    );
}

function diamond(color, size) {
    return new THREE.Mesh(
        new THREE.OctahedronGeometry(size, 0),
        new THREE.MeshStandardMaterial({
            color: hex(color),
            transparent: true,
            opacity: 0.26,
            emissive: hex(color),
            emissiveIntensity: 0.22,
            roughness: 0.22,
            metalness: 0.16,
        }),
    );
}

function curveLine(curve, color, opacity) {
    const points = curve.getPoints(80);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
        color: hex(color),
        transparent: true,
        opacity,
    });
    return new THREE.Line(geometry, material);
}

function straightLine(start, end, color, opacity) {
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    const material = new THREE.LineBasicMaterial({
        color: hex(color),
        transparent: true,
        opacity,
    });
    return new THREE.Line(geometry, material);
}

function pulseSphere(color, radius) {
    return new THREE.Mesh(
        new THREE.SphereGeometry(radius, 18, 18),
        new THREE.MeshStandardMaterial({
            color: hex(color),
            emissive: hex(color),
            emissiveIntensity: 0.9,
            transparent: true,
            opacity: 0.95,
            roughness: 0.12,
            metalness: 0.04,
        }),
    );
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function hex(color) {
    return Number(String(color).replace("#", "0x"));
}
