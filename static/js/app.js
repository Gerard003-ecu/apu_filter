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
                'color': '#475569',
                'font-size': '10px',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': '40px',
                'height': '40px',
                'text-wrap': 'wrap',
                'text-max-width': '80px',
                'border-width': 1,
                'border-color': '#cbd5e1'
            }
        },
        // --- Niveles Jerárquicos ---
        {
            selector: 'node[level=0]', // PROYECTO TOTAL
            style: {
                'background-color': '#1e293b', // Slate 900
                'width': '80px',
                'height': '80px',
                'font-size': '12px',
                'font-weight': 'bold',
                'color': '#ffffff',
                'border-width': 2,
                'border-color': '#0f172a'
            }
        },
        {
            selector: 'node[type="CAPITULO"]', // Nivel 1
            style: {
                'background-color': '#3b82f6', // Blue 500
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
                'color': '#991b1b'
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
                'line-color': '#cbd5e1', // Slate 300
                'target-arrow-color': '#cbd5e1',
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
                'line-color': '#6366f1' // Indigo 500
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
            .replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold mt-2 mb-1 text-slate-800">$1</h3>')
            .replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold mt-3 mb-2 text-slate-900">$1</h2>')
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-slate-900">$1</strong>')
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
            <div class="p-4 rounded-md border ${type === 'error' ? 'bg-red-50 border-red-200 text-red-700' : 'bg-blue-50 border-blue-200 text-blue-700'}">
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

    updateStrategicLevel(audit_report, health_report) {
        const narrativeEl = document.getElementById('strategic-narrative');
        const scoreEl = document.getElementById('viability-score');
        const indicatorEl = document.getElementById('viability-indicator');
        const textEl = document.getElementById('viability-text');

        // Preferir el reporte de auditoría completo si existe, sino fallback al health_report
        if (audit_report || health_report) {
            // Narrativa con Interacción para Evidencia
            const narrativeText = audit_report?.strategic_narrative || health_report?.strategic_narrative;

            if (narrativeText) {
                // Inyectamos un botón/enlace para ver evidencia si la narrativa menciona riesgos
                let html = `<div class="prose prose-sm max-w-none text-slate-600">${Utils.parseMarkdown(narrativeText)}</div>`;

                if (html.includes('Riesgo') || html.includes('Ciclo') || html.includes('Pirámide')) {
                    html += `<div class="mt-3">
                                <button onclick="TopologyController.focusEvidence()" class="text-xs bg-red-50 text-red-600 px-3 py-1 rounded border border-red-200 hover:bg-red-100 transition-colors font-medium flex items-center gap-1">
                                    <span>🔍</span> Ver Evidencia Forense en Grafo
                                </button>
                             </div>`;
                }
                narrativeEl.innerHTML = html;
            } else if (health_report?.executive_report?.circular_risks?.length > 0) {
                 narrativeEl.innerHTML = `<p class="text-red-600 font-bold">⚠️ Se han detectado riesgos estructurales críticos.</p>
                                          <ul class="list-disc pl-5 mt-2 text-sm">${health_report.executive_report.circular_risks.map(r => `<li>${r}</li>`).join('')}</ul>
                                          <button onclick="TopologyController.focusEvidence()" class="mt-2 text-sm text-indigo-600 underline">Ver Nodos Afectados</button>`;
            } else {
                 narrativeEl.innerHTML = `<p>El análisis preliminar indica una estructura estable. Se recomienda revisar las alertas operativas.</p>`;
            }

            // Viabilidad - Prioridad: audit_report.integrity_score, luego fallback
            const integrity = audit_report?.integrity_score || health_report?.business_integrity_score || health_report?.executive_report?.integrity_score || 0;

            if(scoreEl) scoreEl.textContent = `${Math.round(integrity)}/100`;

            // Barra de integridad
            const barEl = document.getElementById('integrity-bar');
            const valEl = document.getElementById('integrity-value');
            if(barEl && valEl) {
                barEl.style.width = `${integrity}%`;
                valEl.textContent = `${Math.round(integrity)}%`;

                // Color dinámico
                if(integrity > 80) barEl.className = "bg-green-600 h-2 rounded-full";
                else if(integrity > 50) barEl.className = "bg-yellow-500 h-2 rounded-full";
                else barEl.className = "bg-red-600 h-2 rounded-full";
            }

            if (integrity >= 80) {
                indicatorEl.className = "h-4 w-4 rounded-full bg-green-500 shadow-lg shadow-green-500/50";
                textEl.textContent = "Proyecto Saludable";
                textEl.className = "text-xs font-bold text-green-600 mt-2";
            } else if (integrity >= 50) {
                indicatorEl.className = "h-4 w-4 rounded-full bg-yellow-500 shadow-lg shadow-yellow-500/50";
                textEl.textContent = "Requiere Atención";
                textEl.className = "text-xs font-bold text-yellow-600 mt-2";
            } else {
                indicatorEl.className = "h-4 w-4 rounded-full bg-red-500 shadow-lg shadow-red-500/50 animate-pulse";
                textEl.textContent = "Riesgo Estructural";
                textEl.className = "text-xs font-bold text-red-600 mt-2";
            }
        }
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
            const response = await fetch(CONFIG.API_ENDPOINTS.TOPOLOGY);
            if (!response.ok) throw new Error('Error obteniendo topología');

            const elements = await response.json();

            if (this.cy) {
                this.cy.destroy();
            }

            // Configuración del Layout Piramidal (Breadthfirst)
            // Se asume que el backend envía 'level' (0=Proyecto, 1=Capítulo, 2=APU, 3=Insumo)
            const layoutConfig = {
                name: 'breadthfirst',
                directed: true,
                padding: 40,
                spacingFactor: 1.5, // Mayor separación vertical
                animate: true,
                animationDuration: 1000,
                roots: 'node[level=0]', // Raíz forzada en la cima
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

            // Ajustar vista inicial
            this.cy.fit();

        } catch (error) {
            console.error("Topology Error:", error);
            container.innerHTML = `<div class="flex items-center justify-center h-full text-red-500">Error visualizando el grafo: ${error.message}</div>`;
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

    // Función para "Hacer Zoom a la Evidencia"
    focusEvidence() {
        if (!this.cy) return;

        // Seleccionar nodos con evidencia o clases de riesgo
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

            // Efecto visual temporal
            const originalBorder = evidenceNodes.style('border-color');
            evidenceNodes.animate({
                style: { 'border-color': '#fbbf24', 'border-width': 10 } // Flash Amber
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
        // Polling cada 5 segundos para actualizar métricas operativas
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
        // Actualizar manómetro de saturación (simulado con métricas reales si existen)
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
            if (voltage > 450) voltText.classList.add('text-red-600');
            else voltText.classList.remove('text-red-600');
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

            // 1. Data Unwrapping & Detection
            // Detectar DataProduct (QFS) y desempaquetar payload si es necesario
            let payload = null;
            let auditReport = null;

            if (result.kind === "DataProduct" && result.payload) {
                console.log("📦 Data Product (QFS) detectado en Frontend");
                payload = result.payload;
            } else if (result.data && result.data.kind === "DataProduct" && result.data.payload) {
                 // Caso donde DataProduct está anidado
                 console.log("📦 Data Product (QFS) anidado detectado");
                 payload = result.data.payload;
            } else {
                 // Fallback legacy
                 payload = result;
            }

            // Extraer reportes de la estructura correcta
            if (payload && payload.audit_report) {
                auditReport = payload.audit_report;
            } else if (result.audit_report) {
                auditReport = result.audit_report;
            }

            // 2. Actualizar Dashboard Estratégico
            // Pasamos tanto el auditReport (QFS) como health_report (Legacy)
            UIManager.updateStrategicLevel(auditReport, result.health_report);

            // 3. Mostrar Contenido Principal
            UIManager.toggleMainContent(true);
            UIManager.showStatus("Topología Generada Exitosamente", 'success');

            // 4. Renderizar Grafo
            // Pequeño delay para asegurar que el DOM es visible
            setTimeout(() => {
                TopologyController.render();
            }, 100);

            // 5. Iniciar Telemetría
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
    // Inicializar HTMX
    if (window.htmx) {
        htmx.logAll();
    }
});
