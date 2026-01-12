document.addEventListener('DOMContentLoaded', () => {
    // === INITIALIZATION ===
    let cy = null;
    let physicsChart = null;

    // === DOM ELEMENTS ===
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');

    // === CHART SETUP ===
    function initCharts() {
        const ctx = document.getElementById('physicsChart').getContext('2d');
        physicsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Saturación',
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        data: [],
                        tension: 0.4
                    },
                    {
                        label: 'Entropía',
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        data: [],
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
                    }
                }
            }
        });
    }

    // === UPLOAD HANDLER ===
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const files = fileInput.files;
        if (files.length === 0) {
            alert("Por favor seleccione archivos para analizar.");
            return;
        }

        // Show Loading
        loadingOverlay.classList.remove('d-none');
        updateLoadingText("Subiendo archivos...");

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        try {
            updateLoadingText("Inyectando datos en Pipeline...");
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Error en la carga');
            }

            const data = await response.json();

            updateLoadingText("Renderizando Caja de Cristal...");
            await renderDashboard(data);

        } catch (error) {
            console.error(error);
            alert(`Error Crítico: ${error.message}`);
        } finally {
            loadingOverlay.classList.add('d-none');
        }
    });

    function updateLoadingText(text) {
        loadingText.textContent = text;
    }

    // === RENDER LOGIC ===
    async function renderDashboard(data) {
        console.log("Datos recibidos:", data);

        // 1. ZONA 0: VEREDICTO
        // Access payload because data structure might be enveloped
        const payload = data.kind === 'DataProduct' ? data.payload : data;

        // Handle Audit Report Structure
        const audit = payload.audit_report || {};
        const narrative = audit.strategic_narrative || "> Análisis pendiente.";

        document.getElementById('strategic-narrative').innerHTML = formatNarrative(narrative);

        // Update Verdict Title based on score if available, else default
        // const score = payload.governance?.compliance_score || 0;
        // document.getElementById('verdict-title').textContent = score > 80 ? "APROBADO POR CONSEJO" : "REVISIÓN REQUERIDA";

        // 2. ZONA 1: FÍSICA
        // Assuming we get telemetry/physics data. For now, mocking or using available fields.
        // In a real scenario, we might poll an endpoint. Here we check the payload.

        // Mock update for visual feedback
        document.getElementById('physics-status').textContent = "ACTIVO";
        document.getElementById('physics-status').classList.add('text-neon-green');

        // 3. ZONA 2: TOPOLOGÍA
        await renderGraph(payload.business_topology_report); // Assuming this key exists or we fetch separate

        // 4. ZONA 3: LOGÍSTICA
        renderBOM(payload.insumos_procesados || []); // Check your API response structure
    }

    function formatNarrative(text) {
        // Simple formatter to treat newlines as <br> and bold markdown
        return text.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>');
    }

    async function renderGraph(topologyReport) {
        // Fetch graph elements from backend if not in payload
        // Or if the payload contains the graph metrics, use them
        // For visualization, we typically hit /api/visualization/project-graph

        try {
            const response = await fetch('/api/visualization/project-graph');
            if (!response.ok) return;
            const elements = await response.json();

            if (cy) cy.destroy();

            cy = cytoscape({
                container: document.getElementById('cy'),
                elements: elements,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'background-color': '#64748b',
                            'label': 'data(label)',
                            'color': '#f1f5f9',
                            'font-size': '10px',
                            'font-family': 'Inter',
                            'text-valign': 'bottom',
                            'text-margin-y': 5
                        }
                    },
                    {
                        selector: 'node[type="APU"]',
                        style: { 'background-color': '#3b82f6', 'width': 30, 'height': 30 }
                    },
                    {
                        selector: 'node[type="INSUMO"]',
                        style: { 'background-color': '#f59e0b', 'width': 20, 'height': 20 }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 1,
                            'line-color': 'rgba(148, 163, 184, 0.2)',
                            'curve-style': 'bezier'
                        }
                    }
                ],
                layout: {
                    name: 'dagre', // Requires cytoscape-dagre
                    rankDir: 'TB'
                }
            });

        } catch (e) {
            console.error("Graph render failed", e);
        }
    }

    function renderBOM(insumos) {
        const tbody = document.getElementById('bom-body');
        const countBadge = document.getElementById('bom-count');

        if (!insumos || insumos.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-4">Sin datos</td></tr>';
            countBadge.textContent = "0 Items";
            return;
        }

        countBadge.textContent = `${insumos.length} Items`;

        // Limit to 100 for performance in preview
        const preview = insumos.slice(0, 100);

        tbody.innerHTML = preview.map(item => `
            <tr>
                <td class="text-neon-blue">${item.CODIGO || item.codigo}</td>
                <td>${(item.DESCRIPCION || item.descripcion || '').substring(0, 50)}...</td>
                <td class="text-muted">${item.UNIDAD || item.unidad}</td>
                <td class="text-end font-monospace text-neon-green">${formatNumber(item.CANTIDAD_TOTAL || item.cantidad_total)}</td>
                <td class="text-end font-monospace">${formatCurrency(item.PRECIO_UNITARIO || item.precio_unitario)}</td>
                <td class="text-end font-monospace text-white">${formatCurrency(item.VALOR_TOTAL || item.valor_total)}</td>
                <td><span class="badge bg-dark border border-secondary text-xs">${item.TIPO || item.tipo || 'N/A'}</span></td>
            </tr>
        `).join('');
    }

    function formatNumber(num) {
        return new Intl.NumberFormat('es-CO', { maximumFractionDigits: 2 }).format(num || 0);
    }

    function formatCurrency(num) {
        return new Intl.NumberFormat('es-CO', { style: 'currency', currency: 'COP' }).format(num || 0);
    }

    // Toggle BOM
    window.toggleBOM = function() {
        const container = document.querySelector('.table-responsive');
        if (container.style.maxHeight === '300px') {
            container.style.maxHeight = 'none';
        } else {
            container.style.maxHeight = '300px';
        }
    };

    // Initialize
    initCharts();
});
