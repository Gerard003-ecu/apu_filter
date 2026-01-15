// ============================================================================
// CONSTANTES Y CONFIGURACIÓN
// ============================================================================
const CONFIG = {
    API_ENDPOINTS: {
        UPLOAD: '/upload',
        ESTIMATE: '/api/estimate',
        APU_DETAIL: '/api/apu/',
        TOPOLOGY: '/api/visualization/project-graph',
        TELEMETRY: '/api/telemetry/status'
    },
    DEBOUNCE_DELAY: 300,
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_FILE_TYPES: ['.csv'],
    CYTOSCAPE_STYLE: [
        // --- Estilos Base ---
        {
            selector: 'node',
            style: {
                'background-color': '#94a3b8',
                'label': 'data(label)',
                'color': '#cbd5e1', // Lighter text for Dark Mode
                'font-size': '10px',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': '40px',
                'height': '40px',
                'text-wrap': 'wrap',
                'text-max-width': '80px',
                'border-width': 1,
                'border-color': '#475569'
            }
        },
        // --- Niveles Jerárquicos ---
        {
            selector: 'node[level=0]', // PROYECTO TOTAL
            style: {
                'background-color': '#3b82f6', // Blue 500
                'width': '80px',
                'height': '80px',
                'font-size': '12px',
                'font-weight': 'bold',
                'color': '#ffffff',
                'border-width': 2,
                'border-color': '#60a5fa'
            }
        },
        {
            selector: 'node[type="CAPITULO"]', // Nivel 1
            style: {
                'background-color': '#6366f1', // Indigo 500
                'shape': 'hexagon',
                'width': '60px',
                'height': '60px'
            }
        },
        {
            selector: 'node[type="APU"]', // Nivel 2
            style: {
                'background-color': '#10b981', // Emerald 500
                'shape': 'round-rectangle',
                'width': '50px',
                'height': '40px'
            }
        },
        {
            selector: 'node[type="INSUMO"]', // Nivel 3
            style: {
                'background-color': '#f97316', // Orange 500
                'shape': 'ellipse',
                'width': '30px',
                'height': '30px',
                'font-size': '8px'
            }
        },

        // --- Estilos Forenses (Caja de Cristal) ---
        {
            selector: '.circular-dependency-node',
            style: {
                'background-color': '#ef4444', // Red 500
                'border-width': '6px',
                'border-color': '#991b1b', // Red 800
                'width': '70px',
                'height': '70px',
                'label': 'data(label)',
                'font-weight': 'bold',
                'color': '#fca5a5'
            }
        },
        {
            selector: '.inverted-pyramid-stress',
            style: {
                'background-color': '#f59e0b', // Amber 500
                'shape': 'diamond',
                // Mapeo dinámico: Mayor peso financiero = Mayor tamaño visual
                // Usamos un rango seguro para evitar nodos infinitos
                'width': 'mapData(weight, 0, 100000000, 60, 150)',
                'height': 'mapData(weight, 0, 100000000, 60, 150)',
                'border-width': '2px',
                'border-color': '#d97706',
                'label': 'data(label)'
            }
        },
        {
            selector: '.isolated',
            style: {
                'background-color': '#e2e8f0', // Slate 200
                'opacity': 0.5,
                'border-style': 'dashed'
            }
        },

        // --- Aristas ---
        {
            selector: 'edge',
            style: {
                'width': 1,
                'line-color': '#475569', // Slate 600
                'target-arrow-color': '#475569',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 1.5
            }
        },
        {
            selector: 'edge[?is_evidence]',
            style: {
                'width': 4,
                'line-color': '#ef4444', // Red 500
                'target-arrow-color': '#ef4444',
                'line-style': 'dashed',
                'target-arrow-shape': 'triangle-backcurve'
            }
        },
        {
            selector: 'edge[cost > 1000000]', // High value paths
            style: {
                'width': 3,
                'line-color': '#818cf8' // Indigo 400
            }
        }
    ]
};

// ============================================================================
// UTILIDADES
// ============================================================================
const Utils = {
    validateFile(file, maxSize = CONFIG.MAX_FILE_SIZE) {
        if (!file) throw new Error('No se ha seleccionado ningún archivo');
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        if (!CONFIG.ALLOWED_FILE_TYPES.includes(extension)) {
            throw new Error(`Tipo de archivo no permitido. Use: ${CONFIG.ALLOWED_FILE_TYPES.join(', ')}`);
        }
        if (file.size > maxSize) {
            throw new Error(`El archivo excede el tamaño máximo permitido.`);
        }
        return true;
    },
    
    // Función de ayuda para formatear moneda
    formatCurrency(value) {
        return new Intl.NumberFormat('es-CO', { style: 'currency', currency: 'COP' }).format(value);
    },

    // Función simple para parsear Markdown básico
    parseMarkdown(text) {
        if (!text) return '';
        let html = text
            // Headers
            .replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold mt-2 mb-1 text-slate-200">$1</h3>')
            .replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold mt-3 mb-2 text-white">$1</h2>')
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>')
            // Lists
            .replace(/^\- (.*$)/gim, '<li class="ml-4 list-disc">$1</li>')
            // Line breaks
            .replace(/\n/g, '<br>');
        return html;
    }
};

// ============================================================================
// GESTOR DE UI
// ============================================================================
const UIManager = {
    toggleLoader(show) {
        const loader = document.getElementById('loader');
        if (loader) loader.classList.toggle('hidden', !show);
    },

    showStatus(message, type = 'info') {
        const statusEl = document.getElementById('status');
        if (!statusEl) return;

        statusEl.innerHTML = `
            <div class="p-4 rounded-md border ${type === 'error' ? 'bg-red-900/50 border-red-700 text-red-200' : 'bg-blue-900/50 border-blue-700 text-blue-200'}">
                ${message}
            </div>
        `;
        statusEl.classList.remove('hidden');

        if (type !== 'error') {
            setTimeout(() => statusEl.classList.add('hidden'), 5000);
        }
    },

    toggleMainContent(show) {
        document.getElementById('upload-container').classList.toggle('hidden', show);
        document.getElementById('main-content').classList.toggle('hidden', !show);
    },

    updateStrategicLevel(payload, result) {
        // Deep QFS Mapping
        // payload: data.payload (The DataProduct)
        // result: data (The raw response)

        const narrativeEl = document.getElementById('strategic-narrative');
        const scoreEl = document.getElementById('viability-score');
        const indicatorEl = document.getElementById('viability-indicator');
        const textEl = document.getElementById('viability-text');

        // Extract Audit Report from QFS structure
        let auditReport = null;
        if (payload && payload.audit_report) {
            auditReport = payload.audit_report;
        } else if (result && result.audit_report) {
             auditReport = result.audit_report;
        }

        // --- 1. Narrativa Estratégica ---
        if (auditReport && auditReport.strategic_narrative) {
            const narrativeText = auditReport.strategic_narrative;
            let html = `<div class="prose prose-sm max-w-none text-slate-300">${Utils.parseMarkdown(narrativeText)}</div>`;

            if (html.includes('Riesgo') || html.includes('Ciclo') || html.includes('Pirámide')) {
                html += `<div class="mt-3">
                            <button onclick="TopologyController.focusEvidence()" class="text-xs bg-red-900/30 text-red-300 px-3 py-1 rounded border border-red-800 hover:bg-red-900/50 transition-colors font-medium flex items-center gap-1">
                                <span>🔍</span> Ver Evidencia Forense en Grafo
                            </button>
                         </div>`;
            }
            narrativeEl.innerHTML = html;
        } else {
             narrativeEl.innerHTML = `<p class="text-slate-400 italic">Esperando análisis...</p>`;
        }

        // --- 2. Integrity Score & Financial Risk ---
        // Access nested metrics safely
        const integrity = auditReport?.integrity_score || 0;
        const financialRisk = auditReport?.financial_risk_level || "DESCONOCIDO";

        // Update Risk Display
        const riskLevelEl = document.getElementById('financial-risk-level');
        if (riskLevelEl) riskLevelEl.textContent = financialRisk;

        // Update Integrity Bar
        const barEl = document.getElementById('integrity-bar');
        const valEl = document.getElementById('integrity-value');
        if(barEl && valEl) {
            barEl.style.width = `${integrity}%`;
            valEl.textContent = `${Math.round(integrity)}%`;

            if(integrity > 80) barEl.className = "bg-green-600 h-2 rounded-full";
            else if(integrity > 50) barEl.className = "bg-yellow-500 h-2 rounded-full";
            else barEl.className = "bg-red-600 h-2 rounded-full";
        }

        // --- 3. Deep Metric Mapping (QFS Paths) ---
        // β0: payload.audit_report.details.topological_invariants.betti_numbers.beta_0
        // β1: payload.audit_report.details.topological_invariants.betti_numbers.beta_1
        // Ψ: payload.audit_report.details.topological_invariants.pyramid_stability
        // WACC: payload.audit_report.details.financial_metrics_input.wacc

        const details = auditReport?.details || {};
        const invariants = details.topological_invariants || {};
        const betti = invariants.betti_numbers || {};
        const financial = details.financial_metrics_input || {};

        const beta0 = betti.beta_0 !== undefined ? betti.beta_0 : '--';
        const beta1 = betti.beta_1 !== undefined ? betti.beta_1 : '--';
        // Stability might be directly under details or inside topological_invariants depending on schema version
        const psi = invariants.pyramid_stability !== undefined
                    ? invariants.pyramid_stability.toFixed(3)
                    : (details.pyramid_stability !== undefined ? details.pyramid_stability.toFixed(3) : '--');

        // Map WACC if element exists
        const waccEl = document.getElementById('val-wacc'); // Assuming ID based on prompt context, though not visible in snippet
        if (waccEl) {
             const wacc = financial.wacc !== undefined ? (financial.wacc * 100).toFixed(2) + '%' : '--';
             waccEl.textContent = wacc;
        }

        document.getElementById('beta0-score').textContent = beta0;
        document.getElementById('beta1-score').textContent = beta1;
        document.getElementById('psi-score').textContent = psi;

        // Visual Feedback based on Score
        if (integrity >= 80) {
            indicatorEl.className = "h-4 w-4 rounded-full bg-green-500 shadow-lg shadow-green-500/50";
            textEl.textContent = "Proyecto Saludable";
            textEl.className = "text-xs font-bold text-green-400 mt-2";
        } else if (integrity >= 50) {
            indicatorEl.className = "h-4 w-4 rounded-full bg-yellow-500 shadow-lg shadow-yellow-500/50";
            textEl.textContent = "Requiere Atención";
            textEl.className = "text-xs font-bold text-yellow-400 mt-2";
        } else {
            indicatorEl.className = "h-4 w-4 rounded-full bg-red-500 shadow-lg shadow-red-500/50 animate-pulse";
            textEl.textContent = "Riesgo Estructural";
            textEl.className = "text-xs font-bold text-red-400 mt-2";
        }
    },

    updateAPUTable(payload) {
        const tableBody = document.getElementById('apu-table-body');
        const countEl = document.getElementById('apu-count');

        if (!tableBody) return;
        tableBody.innerHTML = ''; // Clear existing

        let apus = [];

        // Estrategia de búsqueda robusta para APUs
        if (payload && payload.processed_apus) {
            apus = payload.processed_apus;
        } else if (payload && payload.payload && payload.payload.processed_apus) {
            apus = payload.payload.processed_apus;
        } else if (payload && payload.data && payload.data.processed_apus) {
            apus = payload.data.processed_apus;
        }

        console.log("APUs para tabla:", apus);

        countEl.textContent = `${apus.length} registros`;

        // Limit to first 50 for performance (pagination should be server-side ideally, but client-side for now)
        const displayAPUs = apus.slice(0, 50);

        if (displayAPUs.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="4" class="px-6 py-4 text-center text-slate-500 italic">No se encontraron APUs procesados</td></tr>';
            return;
        }

        displayAPUs.forEach(apu => {
            const row = document.createElement('tr');
            row.className = "hover:bg-slate-800 transition-colors border-b border-slate-700";

            // Safe accessors
            const code = apu.CODIGO_APU || apu.CODIGO || apu.ITEM || apu.id || '--';
            const desc = apu.DESCRIPCION_APU || apu.DESCRIPCION || apu.NOMBRE || 'Sin descripción';
            const unit = apu.UNIDAD_APU || apu.UNIDAD || apu.UND || '--';

            // Mejora la lógica de extracción del costo para buscar múltiples claves posibles
            const cost = apu.COSTO_UNITARIO_TOTAL || apu.PRECIO_UNIT_APU || apu.VALOR_TOTAL || apu.VALOR_TOTAL_APU || apu.valor_total || 0;

            row.innerHTML = `
                <td class="px-6 py-3 text-slate-300 font-mono text-xs">${code}</td>
                <td class="px-6 py-3 text-slate-300 text-xs">${desc}</td>
                <td class="px-6 py-3 text-slate-400 text-xs">${unit}</td>
                <td class="px-6 py-3 text-right text-slate-200 font-mono text-xs">${Utils.formatCurrency(cost)}</td>
            `;
            tableBody.appendChild(row);
        });
    }
};

// ============================================================================
// CONTROLADOR DE TOPOLOGÍA (Cytoscape)
// ============================================================================
const TopologyController = {
    cy: null,

    async render() {
        const container = document.getElementById('cy');
        const loader = document.getElementById('graph-loader');
        if (!container) return;

        loader.classList.remove('hidden');

        try {
            // Explicitly fetch the graph data
            const response = await fetch(CONFIG.API_ENDPOINTS.TOPOLOGY);
            if (!response.ok) throw new Error('Error obteniendo topología');

            // The API returns the Cytoscape elements structure directly or wrapped
            const graphData = await response.json();

            // Ensure elements are in the correct format { elements: [...] } or just [...]
            // API returns a list of elements usually
            let elements = graphData;
            if (graphData.elements) elements = graphData.elements;

            if (this.cy) {
                this.cy.destroy();
            }

            // Configuración del Layout Piramidal (Breadthfirst)
            const layoutConfig = {
                name: 'breadthfirst',
                directed: true,
                padding: 40,
                spacingFactor: 1.5,
                animate: true,
                animationDuration: 1000,
                roots: 'node[level=0]',
                avoidOverlap: true
            };

            this.cy = cytoscape({
                container: container,
                elements: elements,
                style: CONFIG.CYTOSCAPE_STYLE,
                layout: layoutConfig,
                minZoom: 0.1,
                maxZoom: 4,
                wheelSensitivity: 0.2
            });

            this._setupEvents();
            this.cy.fit();

        } catch (error) {
            console.error("Topology Error:", error);
            container.innerHTML = `<div class="flex items-center justify-center h-full text-red-400">Error visualizando el grafo: ${error.message}</div>`;
        } finally {
            loader.classList.add('hidden');
        }
    },

    _setupEvents() {
        if (!this.cy) return;
        
        this.cy.on('tap', 'node', (evt) => {
            const node = evt.target;
            this._showNodeDetails(node);
        });
        
        this.cy.on('tap', (evt) => {
            if (evt.target === this.cy) {
                document.getElementById('node-details').classList.add('hidden');
            }
        });
    },

    _showNodeDetails(node) {
        const data = node.data();
        const detailsPanel = document.getElementById('node-details');

        document.getElementById('nd-label').textContent = data.label ? data.label.replace('\n', ' ') : data.id;
        document.getElementById('nd-type').textContent = data.type;
        document.getElementById('nd-cost').textContent = Utils.formatCurrency(data.cost || 0);

        // Mostrar botón de acción si es necesario
        const actionBtn = document.getElementById('nd-action-btn');
        if (node.hasClass('circular-dependency-node') || node.hasClass('inverted-pyramid-stress')) {
            actionBtn.classList.remove('hidden');
            actionBtn.textContent = "Ver Análisis de Impacto";
            actionBtn.onclick = () => alert(`Análisis para ${data.id}: \nCosto: ${Utils.formatCurrency(data.cost)}\nNivel de Estrés: ALTO`);
        } else {
            actionBtn.classList.add('hidden');
        }

        detailsPanel.classList.remove('hidden');
    },

    focusEvidence() {
        if (!this.cy) return;
        const evidenceNodes = this.cy.elements('node[?is_evidence], .circular-dependency-node, .inverted-pyramid-stress');

        if (evidenceNodes.length > 0) {
            this.cy.animate({
                fit: {
                    eles: evidenceNodes,
                    padding: 80
                },
                duration: 1200,
                easing: 'ease-in-out-cubic'
            });

            const originalBorder = evidenceNodes.style('border-color');
            evidenceNodes.animate({
                style: { 'border-color': '#fbbf24', 'border-width': 10 }
            }, {
                duration: 500,
                complete: () => {
                    evidenceNodes.animate({
                         style: { 'border-color': originalBorder, 'border-width': 6 }
                    }, { duration: 500 });
                }
            });

        } else {
            UIManager.showStatus("No hay evidencia forense crítica visible en este momento.", "info");
        }
    }
};

// ============================================================================
// CONTROLADOR OPERATIVO (Telemetría)
// ============================================================================
const OperationsController = {
    startPolling() {
        setInterval(async () => {
            try {
                const response = await fetch(CONFIG.API_ENDPOINTS.TELEMETRY);
                if (response.ok) {
                    const status = await response.json();
                    this._updateDashboard(status);
                }
            } catch (e) {
                console.error("Telemetry poll failed", e);
            }
        }, 5000);
    },

    _updateDashboard(status) {
        const saturation = status.flux_condenser?.avg_saturation || 0;
        const voltage = status.flux_condenser?.max_flyback_voltage || 0;

        const bar = document.getElementById('flux-bar');
        const text = document.getElementById('flux-saturation');
        const voltText = document.getElementById('flyback-voltage');

        if (bar && text) {
            bar.style.width = `${Math.min(100, saturation * 100)}%`;
            text.textContent = `${(saturation * 100).toFixed(1)}%`;

            if (saturation > 0.8) bar.classList.replace('bg-blue-600', 'bg-red-600');
            else bar.classList.replace('bg-red-600', 'bg-blue-600');
        }

        if (voltText) {
            voltText.textContent = `${voltage.toFixed(1)}V`;
            if (voltage > 450) voltText.classList.add('text-red-500');
            else voltText.classList.remove('text-red-500');
        }
    }
};

// ============================================================================
// CONTROLADOR PRINCIPAL
// ============================================================================
const AppController = {
    async procesarArchivos() {
        const presupuesto = document.getElementById('presupuesto_file')?.files[0];
        const apus = document.getElementById('apus_file')?.files[0];
        const insumos = document.getElementById('insumos_file')?.files[0];

        try {
            Utils.validateFile(presupuesto);
            Utils.validateFile(apus);
            Utils.validateFile(insumos);
        } catch (error) {
            UIManager.showStatus(error.message, 'error');
            return;
        }

        const formData = new FormData();
        formData.append('presupuesto', presupuesto);
        formData.append('apus', apus);
        formData.append('insumos', insumos);

        UIManager.toggleLoader(true);
        UIManager.showStatus("Iniciando procesamiento de alta velocidad...", 'info');

        try {
            const response = await fetch(`${CONFIG.API_ENDPOINTS.UPLOAD}?include_data=true`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(await response.text());

            const result = await response.json();

            // --- QFS Handling Strategy ---
            let payload = null;

            // Prioritize inner data first as per app.py logic which unwraps payloads
            if (result.data) {
                if (result.data.kind === "DataProduct" && result.data.payload) {
                    console.log("📦 Nested Data Product detected");
                    payload = result.data.payload;
                } else {
                     // Direct data or unwrapped payload
                     console.log("⚠️ Direct/Unwrapped data detected");
                     payload = result.data;
                }
            } else if (result.kind === "DataProduct" && result.payload) {
                console.log("📦 Data Product (QFS) detected at root");
                payload = result.payload;
            } else {
                 console.warn("⚠️ Legacy format detected - attempting best effort mapping");
                 payload = result;
            }

            // 1. Update Strategic Dashboard with QFS payload
            UIManager.updateStrategicLevel(payload, result);

            // 2. Update APU Table
            // Pass the derived payload, but also fallback to checking result if needed
            // The updateAPUTable function now has robust checking internally too.
            UIManager.updateAPUTable(payload);

            // 3. Show Content
            UIManager.toggleMainContent(true);
            UIManager.showStatus("Topología Generada Exitosamente", 'success');

            // 4. Render Topology
            setTimeout(() => {
                TopologyController.render();
            }, 100);

            // 5. Start Telemetry
            OperationsController.startPolling();

        } catch (error) {
            console.error(error);
            UIManager.showStatus(`Error: ${error.message}`, 'error');
        } finally {
            UIManager.toggleLoader(false);
        }
    }
};

// ============================================================================
// INICIALIZACIÓN
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log("Sistema MIC v2.0 Inicializado");
    if (window.htmx) {
        htmx.logAll();
    }
});
