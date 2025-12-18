// ============================================================================
// CONSTANTES Y CONFIGURACIÓN
// ============================================================================
const CONFIG = {
    API_ENDPOINTS: {
        UPLOAD: '/upload',
        ESTIMATE: '/api/estimate',
        APU_DETAIL: '/api/apu/',
        TOPOLOGY: '/api/visualization/topology',
        TELEMETRY: '/api/telemetry/status'
    },
    DEBOUNCE_DELAY: 300,
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_FILE_TYPES: ['.csv'],
    CYTOSCAPE_STYLE: [
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
                'text-max-width': '80px'
            }
        },
        {
            selector: 'node[type="ROOT"]',
            style: {
                'background-color': '#4f46e5',
                'width': '60px',
                'height': '60px',
                'font-weight': 'bold',
                'color': '#ffffff'
            }
        },
        {
            selector: 'node[type="CAPITULO"]',
            style: {
                'background-color': '#0ea5e9',
                'shape': 'hexagon',
                'width': '50px',
                'height': '50px'
            }
        },
        {
            selector: 'node[type="APU"]',
            style: {
                'background-color': '#10b981',
                'shape': 'round-rectangle',
                'width': '40px',
                'height': '30px'
            }
        },
        {
            selector: 'node[type="INSUMO"]',
            style: {
                'background-color': '#f59e0b',
                'shape': 'ellipse',
                'width': '20px',
                'height': '20px',
                'font-size': '8px'
            }
        },
        {
            selector: 'node.cycle',
            style: {
                'background-color': '#ef4444',
                'border-width': 3,
                'border-color': '#7f1d1d',
                'label': 'data(label)\n(CICLO)'
            }
        },
        {
            selector: 'node.isolated',
            style: {
                'background-color': '#cbd5e1',
                'opacity': 0.6
            }
        },
        {
            selector: 'edge',
            style: {
                'width': 1,
                'line-color': '#cbd5e1',
                'target-arrow-color': '#cbd5e1',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier'
            }
        },
        {
            selector: 'edge[cost > 1000000]',
            style: {
                'width': 3,
                'line-color': '#6366f1',
                'target-arrow-color': '#6366f1'
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

    updateStrategicLevel(report) {
        const narrativeEl = document.getElementById('strategic-narrative');
        const scoreEl = document.getElementById('viability-score');
        const indicatorEl = document.getElementById('viability-indicator');
        const textEl = document.getElementById('viability-text');

        if (report) {
            // Narrativa
            if (report.strategic_narrative) {
                narrativeEl.innerHTML = `<p>${report.strategic_narrative}</p>`;
            } else if (report.executive_report?.circular_risks?.length > 0) {
                 narrativeEl.innerHTML = `<p class="text-red-600 font-bold">⚠️ Se han detectado riesgos estructurales críticos.</p>
                                          <ul class="list-disc pl-5 mt-2 text-sm">${report.executive_report.circular_risks.map(r => `<li>${r}</li>`).join('')}</ul>`;
            } else {
                 narrativeEl.innerHTML = `<p>El análisis preliminar indica una estructura estable. Se recomienda revisar las alertas operativas.</p>`;
            }

            // Viabilidad
            const integrity = report.business_integrity_score || report.executive_report?.integrity_score || 0;
            scoreEl.textContent = `${Math.round(integrity)}/100`;

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

            this.cy = cytoscape({
                container: container,
                elements: elements,
                style: CONFIG.CYTOSCAPE_STYLE,
                layout: {
                    name: 'cose',
                    animate: false,
                    nodeDimensionsIncludeLabels: true,
                    idealEdgeLength: 100,
                    edgeElasticity: 0.45,
                    nestingFactor: 0.1,
                },
                minZoom: 0.2,
                maxZoom: 3,
            });

            this._setupEvents();

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

        document.getElementById('nd-label').textContent = data.label.replace('\n', ' ');
        document.getElementById('nd-type').textContent = data.type;
        document.getElementById('nd-cost').textContent = Utils.formatCurrency(data.cost || 0);

        // Mostrar botón de acción si es necesario
        const actionBtn = document.getElementById('nd-action-btn');
        if (node.hasClass('cycle') || node.hasClass('isolated')) {
            actionBtn.classList.remove('hidden');
            actionBtn.onclick = () => alert(`Acción correctiva iniciada para ${data.id}`);
        } else {
            actionBtn.classList.add('hidden');
        }

        detailsPanel.classList.remove('hidden');
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

            // 1. Actualizar Dashboard Estratégico
            UIManager.updateStrategicLevel(result.health_report);

            // 2. Mostrar Contenido Principal
            UIManager.toggleMainContent(true);
            UIManager.showStatus("Topología Generada Exitosamente", 'success');

            // 3. Renderizar Grafo
            // Pequeño delay para asegurar que el DOM es visible
            setTimeout(() => {
                TopologyController.render();
            }, 100);

            // 4. Iniciar Telemetría
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
