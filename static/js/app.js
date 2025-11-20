// ============================================================================
// CONSTANTES Y CONFIGURACIÓN
// ============================================================================
const CONFIG = {
    API_ENDPOINTS: {
        UPLOAD: '/upload',
        ESTIMATE: '/api/estimate',
        APU_DETAIL: '/api/apu/'
    },
    DEBOUNCE_DELAY: 300,
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_FILE_TYPES: ['.csv'],
    AIU_DEFAULTS: {
        admin: 9,
        imprev: 3,
        util: 5
    },
    AIU_LIMITS: {
        min: 0,
        max: 100
    },
    MATERIAL_MAP: {
        'CUBIERTA': ['TST', 'PANEL SANDWICH'],
        'FACHADA': ['PANEL SANDWICH', 'LOUVER', 'PERFORADA']
    },
    
    // ✨ NUEVO: Mapeo Backend → Frontend para categorías de insumos
    CATEGORY_TRANSLATION: {
        // Clave del backend: Etiqueta para mostrar al usuario
        'SUMINISTRO': 'MATERIALES',
        'MANO_DE_OBRA': 'MANO DE OBRA',
        'EQUIPO': 'EQUIPO',
        'TRANSPORTE': 'TRANSPORTE',
        'HERRAMIENTA': 'HERRAMIENTAS',
        'OTROS': 'OTROS'
    },
    
    // ✨ NUEVO: Orden de visualización de categorías en el modal
    CATEGORY_DISPLAY_ORDER: [
        'MATERIALES',      // SUMINISTRO traducido
        'MANO DE OBRA',    // MANO_DE_OBRA traducido
        'EQUIPO',
        'TRANSPORTE',
        'HERRAMIENTAS',
        'OTROS'
    ],
    
    APU_TYPE_ORDER: {
        'Suministro': 1,
        'Suministro (Pre-fabricado)': 2,
        'Instalación': 3,
        'Obra Completa': 4,
        'Indefinido': 99
    },
    CURRENCY_OPTIONS: {
        style: 'currency',
        currency: 'COP',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }
};

// ============================================================================
// UTILIDADES
// ============================================================================
const Utils = {
    /**
     * Debounce function para limitar llamadas
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Formatea valores monetarios de forma segura
     */
    formatCurrency(value) {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) return '$0';
        return numValue.toLocaleString('es-CO', CONFIG.CURRENCY_OPTIONS);
    },

    /**
     * Valida y sanitiza números
     */
    sanitizeNumber(value, defaultValue = 0, min = -Infinity, max = Infinity) {
        const num = parseFloat(value);
        if (isNaN(num)) return defaultValue;
        return Math.max(min, Math.min(max, num));
    },

    /**
     * Valida estructura de datos recibidos
     */
    validateDataStructure(data, requiredFields) {
        if (!data || typeof data !== 'object') {
            throw new Error('Datos inválidos: no es un objeto');
        }

        const missingFields = requiredFields.filter(field => !(field in data));
        if (missingFields.length > 0) {
            throw new Error(`Campos faltantes: ${missingFields.join(', ')}`);
        }

        return true;
    },

    /**
     * Validación de archivos
     */
    validateFile(file, maxSize = CONFIG.MAX_FILE_SIZE) {
        if (!file) {
            throw new Error('No se ha seleccionado ningún archivo');
        }

        const extension = '.' + file.name.split('.').pop().toLowerCase();
        if (!CONFIG.ALLOWED_FILE_TYPES.includes(extension)) {
            throw new Error(`Tipo de archivo no permitido. Use: ${CONFIG.ALLOWED_FILE_TYPES.join(', ')}`);
        }

        if (file.size > maxSize) {
            const maxSizeMB = (maxSize / (1024 * 1024)).toFixed(2);
            throw new Error(`El archivo excede el tamaño máximo de ${maxSizeMB}MB`);
        }

        return true;
    },

    /**
     * Deep clone seguro
     */
    deepClone(obj) {
        try {
            return JSON.parse(JSON.stringify(obj));
        } catch (error) {
            Logger.error('Error en deep clone:', error);
            return obj;
        }
    },

    /**
     * Escape HTML para prevenir XSS
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, m => map[m]);
    },
    
    /**
     * ✨ NUEVO: Traduce clave del backend a etiqueta del frontend
     */
    translateCategory(backendKey) {
        // Normalizar la clave (uppercase, trim)
        const normalizedKey = String(backendKey).trim().toUpperCase();
        
        // Buscar traducción exacta
        if (CONFIG.CATEGORY_TRANSLATION[normalizedKey]) {
            return CONFIG.CATEGORY_TRANSLATION[normalizedKey];
        }
        
        // Fallback: intentar buscar por coincidencia parcial
        for (const [key, label] of Object.entries(CONFIG.CATEGORY_TRANSLATION)) {
            if (normalizedKey.includes(key) || key.includes(normalizedKey)) {
                Logger.warn(`Traducción por coincidencia parcial: ${backendKey} → ${label}`);
                return label;
            }
        }
        
        // Si no hay traducción, devolver la clave original capitalizada
        Logger.warn(`Categoría desconocida sin traducción: ${backendKey}`);
        return this.capitalizeWords(normalizedKey);
    },
    
    /**
     * ✨ NUEVO: Capitaliza palabras para etiquetas desconocidas
     */
    capitalizeWords(str) {
        return str
            .toLowerCase()
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
};

// ============================================================================
// SISTEMA DE LOGGING
// ============================================================================
const Logger = {
    _logs: [],
    _maxLogs: 100,

    log(message, data = null) {
        const entry = { level: 'INFO', message, data, timestamp: new Date().toISOString() };
        console.log(`[INFO] ${message}`, data || '');
        this._addLog(entry);
    },

    warn(message, data = null) {
        const entry = { level: 'WARN', message, data, timestamp: new Date().toISOString() };
        console.warn(`[WARN] ${message}`, data || '');
        this._addLog(entry);
    },

    error(message, error = null) {
        const entry = {
            level: 'ERROR',
            message,
            error: error ? { message: error.message, stack: error.stack } : null,
            timestamp: new Date().toISOString()
        };
        console.error(`[ERROR] ${message}`, error || '');
        this._addLog(entry);
    },

    _addLog(entry) {
        this._logs.push(entry);
        if (this._logs.length > this._maxLogs) {
            this._logs.shift();
        }
    },

    getLogs(level = null) {
        return level ? this._logs.filter(log => log.level === level) : this._logs;
    },

    exportLogs() {
        return JSON.stringify(this._logs, null, 2);
    }
};

// ============================================================================
// GESTOR DE ESTADO
// ============================================================================
const StateManager = {
    _state: {
        rawData: {},
        budgetData: [],
        currentTab: 'simulador',
        aiuPercentages: { ...CONFIG.AIU_DEFAULTS },
        isProcessing: false,
        estimatorParams: {}
    },
    _listeners: [],

    /**
     * Obtener estado completo o parcial
     */
    getState(key = null) {
        return key ? this._state[key] : { ...this._state };
    },

    /**
     * Actualizar estado con validación
     */
    setState(updates) {
        const previousState = { ...this._state };
        this._state = { ...this._state, ...updates };

        Logger.log('Estado actualizado', { updates });
        this._notifyListeners(previousState, this._state);
    },

    /**
     * Suscribirse a cambios de estado
     */
    subscribe(listener) {
        if (typeof listener === 'function') {
            this._listeners.push(listener);
        }
    },

    /**
     * Notificar a los listeners
     */
    _notifyListeners(prevState, newState) {
        this._listeners.forEach(listener => {
            try {
                listener(newState, prevState);
            } catch (error) {
                Logger.error('Error en listener de estado:', error);
            }
        });
    },

    /**
     * Resetear estado
     */
    reset() {
        this._state = {
            rawData: {},
            budgetData: [],
            currentTab: 'simulador',
            aiuPercentages: { ...CONFIG.AIU_DEFAULTS },
            isProcessing: false,
            estimatorParams: {}
        };
        Logger.log('Estado reseteado');
    }
};

// ============================================================================
// GESTOR DE UI
// ============================================================================
const UIManager = {
    /**
     * Mostrar/ocultar loader
     */
    toggleLoader(show) {
        const loader = document.getElementById('loader');
        if (loader) {
            loader.classList.toggle('hidden', !show);
        }
    },

    /**
     * Mostrar mensajes de estado
     */
    showStatus(message, type = 'info') {
        const statusEl = document.getElementById('status');
        if (!statusEl) return;

        const classes = {
            info: 'bg-blue-100 text-blue-700 border-blue-300',
            success: 'bg-green-100 text-green-700 border-green-300',
            error: 'bg-red-100 text-red-700 border-red-300',
            warning: 'bg-yellow-100 text-yellow-700 border-yellow-300'
        };

        statusEl.textContent = message;
        statusEl.className = `text-center p-4 rounded-md mt-4 border fade-in ${classes[type] || classes.info}`;

        // Auto-hide después de 5 segundos para mensajes de éxito
        if (type === 'success') {
            setTimeout(() => {
                statusEl.classList.add('hidden');
            }, 5000);
        }
    },

    /**
     * Habilitar/deshabilitar botón de procesamiento
     */
    toggleProcessButton(enabled) {
        const btn = document.getElementById('process-btn');
        if (btn) {
            btn.disabled = !enabled;
        }
    },

    /**
     * Actualizar indicadores de archivos
     */
    updateFileStatus(fileId, status, message) {
        const statusEl = document.getElementById(`${fileId}-status`);
        if (statusEl) {
            const icons = {
                success: '✓',
                error: '✗',
                info: 'ℹ'
            };
            statusEl.textContent = `${icons[status] || ''} ${message}`;
            statusEl.className = `text-xs mt-1 ${
                status === 'success' ? 'text-green-600' :
                status === 'error' ? 'text-red-600' :
                'text-gray-500'
            }`;
        }
    },

    /**
     * Mostrar/ocultar contenido principal
     */
    toggleMainContent(show) {
        const uploadContainer = document.getElementById('upload-container');
        const mainContent = document.getElementById('main-content');

        if (uploadContainer && mainContent) {
            uploadContainer.classList.toggle('hidden', show);
            mainContent.classList.toggle('hidden', !show);
        }
    },

    /**
     * Toggle del log del estimador
     */
    toggleEstimatorLog() {
        const logTextarea = document.getElementById('estimator-log');
        const toggleText = document.getElementById('log-toggle-text');

        if (logTextarea && toggleText) {
            const isHidden = logTextarea.classList.contains('hidden');
            logTextarea.classList.toggle('hidden');
            toggleText.textContent = isHidden ? '▼ Ocultar Log de Cálculo' : '▶ Mostrar Log de Cálculo';
        }
    },

    /**
     * Añadir animación de error
     */
    shakeElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.add('error-shake');
            setTimeout(() => element.classList.remove('error-shake'), 500);
        }
    }
};

// ============================================================================
// GESTOR DE MODAL
// ============================================================================
const ModalManager = {
    /**
     * Abrir modal con datos de APU
     */
    async open(apuCode) {
        const modal = document.getElementById('apu-modal');
        const modalBody = document.getElementById('modal-body');
        const modalTitle = document.getElementById('modal-title');

        if (!modal || !modalBody || !modalTitle) {
            Logger.error('Elementos del modal no encontrados');
            return;
        }

        modalTitle.textContent = `Detalle APU: ${Utils.escapeHtml(apuCode)}`;
        modalBody.innerHTML = '<div class="text-center py-8"><div class="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-12 w-12 mx-auto"></div><p class="mt-4 text-gray-600">Cargando datos...</p></div>';

        modal.classList.remove('hidden');
        modal.classList.add('flex');

        try {
            const response = await fetch(`${CONFIG.API_ENDPOINTS.APU_DETAIL}${encodeURIComponent(apuCode)}`, {
                credentials: 'include',
                headers: {
                    'Accept': 'application/json'
                }
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Error ${response.status}: ${errorText}`);
            }

            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('La respuesta del servidor no es JSON válido');
            }

            const data = await response.json();

            // Validar estructura de datos
            Utils.validateDataStructure(data, ['desglose']);

            modalBody.innerHTML = this._buildModalContent(data);
            Logger.log('Modal de APU abierto exitosamente', { apuCode });

        } catch (error) {
            Logger.error('Error al cargar datos del APU', error);
            modalBody.innerHTML = `
                <div class="text-center py-8">
                    <p class="text-red-500 font-semibold">Error al cargar datos</p>
                    <p class="text-gray-600 mt-2">${Utils.escapeHtml(error.message)}</p>
                    <button onclick="ModalManager.close()" class="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                        Cerrar
                    </button>
                </div>
            `;
        }
    },

    /**
     * Cerrar modal
     */
    close() {
        const modal = document.getElementById('apu-modal');
        if (modal) {
            modal.classList.add('hidden');
            modal.classList.remove('flex');
            Logger.log('Modal cerrado');
        }
    },

    /**
     * ✨ REFACTORIZADO: Construir contenido del modal con traducción y ANÁLISIS DE DISCREPANCIA
     */
    _buildModalContent(data) {
        const aiuPcts = StateManager.getState('aiuPercentages');
        let costoDirecto = 0;
        const categoriesBackend = data.desglose || {};
        
        // 1. Traducir y Agrupar (Igual que antes)
        const categoriesTranslated = {};
        for (const [backendKey, items] of Object.entries(categoriesBackend)) {
            if (!Array.isArray(items) || items.length === 0) continue;
            const frontendLabel = Utils.translateCategory(backendKey);
            categoriesTranslated[frontendLabel] = { items: items, backendKey: backendKey };
        }
        
        // 2. Ordenar (Igual que antes)
        const sortedCategories = Object.keys(categoriesTranslated).sort((a, b) => {
            const indexA = CONFIG.CATEGORY_DISPLAY_ORDER.indexOf(a);
            const indexB = CONFIG.CATEGORY_DISPLAY_ORDER.indexOf(b);
            if (indexA !== -1 && indexB !== -1) return indexA - indexB;
            if (indexA !== -1) return -1;
            if (indexB !== -1) return 1;
            return a.localeCompare(b);
        });
        
        let contentHTML = '<div class="space-y-6">';

        // 3. Renderizar Tablas (Igual que antes)
        sortedCategories.forEach(categoryLabel => {
            const categoryData = categoriesTranslated[categoryLabel];
            const categoryItems = categoryData.items;
            let categorySubtotal = 0;
            const isManoObra = categoryLabel.toUpperCase().includes('MANO') || categoryLabel.toUpperCase().includes('OBRA');

            const tableHeaders = isManoObra ? `
                <th scope="col" class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Descripción</th>
                <th scope="col" class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Jornal Total</th>
                <th scope="col" class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Rendimiento</th>
                <th scope="col" class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Valor Total</th>
            ` : `
                <th scope="col" class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Descripción</th>
                <th scope="col" class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Cantidad</th>
                <th scope="col" class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Vr. Unitario</th>
                <th scope="col" class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Valor Total</th>
            `;

            let tableRows = '';
            categoryItems.forEach(item => {
                const vrTotal = Utils.sanitizeNumber(item.valor_total, 0);
                categorySubtotal += vrTotal;
                const alertaIcon = item.alerta ? ` <span class="text-yellow-500" title="${Utils.escapeHtml(item.alerta)}">⚠️</span>` : '';
                const descripcion = Utils.escapeHtml(item.descripcion || 'Sin descripción');

                if (isManoObra) {
                    const jornalTotal = Utils.formatCurrency(item.valor_unitario || 0);
                    const rendimiento = Utils.sanitizeNumber(item.RENDIMIENTO || item.rendimiento, 0).toFixed(2);
                    tableRows += `<tr class="text-sm hover:bg-gray-50"><td class="py-2 px-4">${descripcion}${alertaIcon}</td><td class="py-2 px-4 text-right">${jornalTotal}</td><td class="py-2 px-4 text-right">${rendimiento}</td><td class="py-2 px-4 text-right">${Utils.formatCurrency(vrTotal)}</td></tr>`;
                } else {
                    const cantidad = Utils.sanitizeNumber(item.cantidad, 0).toFixed(3);
                    const vrUnitario = Utils.formatCurrency(item.valor_unitario || 0);
                    tableRows += `<tr class="text-sm hover:bg-gray-50"><td class="py-2 px-4">${descripcion}${alertaIcon}</td><td class="py-2 px-4 text-right">${cantidad}</td><td class="py-2 px-4 text-right">${vrUnitario}</td><td class="py-2 px-4 text-right">${Utils.formatCurrency(vrTotal)}</td></tr>`;
                }
            });

            costoDirecto += categorySubtotal;
            contentHTML += `<div class="border rounded-lg overflow-hidden"><h4 class="bg-gray-100 px-4 py-2 font-semibold text-gray-700 flex items-center justify-between"><span>${categoryLabel}</span><span class="text-xs text-gray-500 font-normal">${categoryItems.length} ítem(s)</span></h4><div class="overflow-x-auto"><table class="min-w-full"><thead class="bg-gray-50"><tr>${tableHeaders}</tr></thead><tbody class="divide-y divide-gray-200">${tableRows}</tbody><tfoot class="bg-gray-100"><tr><td colspan="3" class="py-2 px-4 text-right font-semibold">SUBTOTAL ${categoryLabel}</td><td class="py-2 px-4 text-right font-semibold">${Utils.formatCurrency(categorySubtotal)}</td></tr></tfoot></table></div></div>`;
        });

        // 4. Tabla de Resumen AIU (Igual que antes)
        const adminValue = costoDirecto * (aiuPcts.admin / 100);
        const imprevValue = costoDirecto * (aiuPcts.imprev / 100);
        const utilValue = costoDirecto * (aiuPcts.util / 100);
        const costoTotalAIU = costoDirecto + adminValue + imprevValue + utilValue;

        contentHTML += `<div class="border rounded-lg mt-6"><table class="min-w-full text-sm"><tbody class="divide-y divide-gray-200"><tr class="hover:bg-gray-50"><td class="px-4 py-2 font-semibold text-right">COSTO DIRECTO</td><td class="px-4 py-2 text-right font-semibold">${Utils.formatCurrency(costoDirecto)}</td></tr><tr class="hover:bg-gray-50"><td class="px-4 py-2 text-right text-gray-600">ADMINISTRACIÓN (${aiuPcts.admin}%)</td><td class="px-4 py-2 text-right">${Utils.formatCurrency(adminValue)}</td></tr><tr class="hover:bg-gray-50"><td class="px-4 py-2 text-right text-gray-600">IMPREVISTOS (${aiuPcts.imprev}%)</td><td class="px-4 py-2 text-right">${Utils.formatCurrency(imprevValue)}</td></tr><tr class="hover:bg-gray-50"><td class="px-4 py-2 text-right text-gray-600">UTILIDAD (${aiuPcts.util}%)</td><td class="px-4 py-2 text-right">${Utils.formatCurrency(utilValue)}</td></tr><tr class="bg-blue-100"><td class="px-4 py-2 text-right font-bold text-blue-800 text-base">VALOR TOTAL APU</td><td class="px-4 py-2 text-right font-bold text-blue-800 text-base">${Utils.formatCurrency(costoTotalAIU)}</td></tr></tbody></table></div>`;

        // 5. Simulación Monte Carlo con CAJA DE INSIGHTS
        if (data.simulation) {
            const sim = data.simulation;

            // --- LÓGICA DE INSIGHTS ---
            const diff = sim.mean - costoDirecto;
            const diffPercent = costoDirecto > 0 ? (diff / costoDirecto) * 100 : 0;
            const discardedItems = sim.metadata?.discarded_items || 0;

            let insightHTML = '';
            let insightClass = '';
            let insightIcon = '';

            if (Math.abs(diffPercent) < 1.0) {
                // Diferencia despreciable (< 1%)
                insightClass = 'bg-green-50 text-green-800 border-green-200';
                insightIcon = '✅';
                insightHTML = `<strong>Consistencia Alta:</strong> El modelo probabilístico valida el costo directo.`;
            } else if (diff > 0) {
                // Monte Carlo > Costo Directo (Riesgo)
                insightClass = 'bg-orange-50 text-orange-800 border-orange-200';
                insightIcon = '📈';
                insightHTML = `<strong>Alerta de Riesgo:</strong> El modelo proyecta un sobrecosto probable del <strong>${diffPercent.toFixed(1)}%</strong> debido a la volatilidad histórica de los insumos.`;
            } else {
                // Monte Carlo < Costo Directo (Ahorro o Exclusión)
                insightClass = 'bg-blue-50 text-blue-800 border-blue-200';
                insightIcon = '📉';

                if (discardedItems > 0) {
                    insightHTML = `<strong>Análisis de Datos:</strong> El modelo es menor porque <strong>${discardedItems} ítems</strong> atípicos o con valores extremos fueron excluidos para normalizar la estadística.`;
                } else {
                    insightHTML = `<strong>Proyección Favorable:</strong> El modelo sugiere una posible optimización del <strong>${Math.abs(diffPercent).toFixed(1)}%</strong> en un escenario conservador.`;
                }
            }

            contentHTML += `
                <div class="border rounded-lg mt-6 p-4 bg-gray-50">
                    <h4 class="font-semibold text-center mb-4 text-gray-700">Análisis de Riesgo (Simulación Monte Carlo)</h4>

                    <div class="grid grid-cols-3 gap-4 text-center text-sm mb-4">
                        <div class="p-2 bg-white rounded shadow-sm">
                            <p class="text-gray-500 text-xs uppercase tracking-wide">Costo Esperado (Media)</p>
                            <p class="font-bold text-lg text-gray-800">${Utils.formatCurrency(sim.mean)}</p>
                        </div>
                        <div class="p-2 bg-white rounded shadow-sm">
                            <p class="text-gray-500 text-xs uppercase tracking-wide">Riesgo (Desv. Estándar)</p>
                            <p class="font-bold text-lg text-gray-800">${Utils.formatCurrency(sim.std_dev)}</p>
                        </div>
                        <div class="p-2 bg-white rounded shadow-sm">
                            <p class="text-gray-500 text-xs uppercase tracking-wide">Rango Confianza (90%)</p>
                            <p class="font-bold text-sm text-gray-800 mt-1">${Utils.formatCurrency(sim.percentile_5)} - ${Utils.formatCurrency(sim.percentile_95)}</p>
                        </div>
                    </div>

                    <!-- CAJA DE INSIGHTS -->
                    <div class="border-l-4 p-3 text-sm ${insightClass} rounded-r">
                        <div class="flex items-start">
                            <span class="text-lg mr-2">${insightIcon}</span>
                            <div>${insightHTML}</div>
                        </div>
                    </div>
                </div>
            `;
        }

        return contentHTML + '</div>';
    }
};

// ============================================================================
// CONTROLADOR DE TABLAS
// ============================================================================
const TableController = {
    /**
     * Actualizar tabla del simulador con jerarquía de tres niveles
     */
    updateSimulatorTable() {
        const tableBody = document.getElementById('presupuesto-table-body');
        if (!tableBody) return;

        const budgetData = StateManager.getState('budgetData');
        const aiuPercentages = StateManager.getState('aiuPercentages');
        const aiuFactor = 1 + ((aiuPercentages.admin + aiuPercentages.imprev + aiuPercentages.util) / 100);

        let totalConsolidado = 0;
        tableBody.innerHTML = '';

        // Calcular total consolidado
        budgetData.forEach(item => {
            const valorConstruccionAIU = Utils.sanitizeNumber(item.VALOR_CONSTRUCCION_UN, 0) * aiuFactor;
            totalConsolidado += valorConstruccionAIU * Utils.sanitizeNumber(item.CANTIDAD_PRESUPUESTO, 0);
        });

        // Actualizar display de total
        const totalCostEl = document.getElementById('total-cost');
        if (totalCostEl) {
            totalCostEl.textContent = Utils.formatCurrency(totalConsolidado);
        }

        // Agrupación por tipo de APU
        const apuTypeGroups = this._groupByApuType(budgetData, aiuFactor);
        const sortedApuTypes = Object.keys(apuTypeGroups).sort((a, b) =>
            (CONFIG.APU_TYPE_ORDER[a] || 99) - (CONFIG.APU_TYPE_ORDER[b] || 99)
        );

        // Renderizar jerarquía
        sortedApuTypes.forEach(apuType => {
            this._renderApuTypeGroup(tableBody, apuType, apuTypeGroups[apuType], aiuFactor);
        });
    },

    /**
     * Agrupar por tipo de APU
     */
    _groupByApuType(budgetData, aiuFactor) {
        return budgetData.reduce((acc, item) => {
            const type = item.tipo_apu || 'Indefinido';
            if (!acc[type]) {
                acc[type] = { items: [], subtotal: 0 };
            }
            acc[type].items.push(item);
            const itemCost = Utils.sanitizeNumber(item.VALOR_CONSTRUCCION_UN, 0) *
                           Utils.sanitizeNumber(item.CANTIDAD_PRESUPUESTO, 0) * aiuFactor;
            acc[type].subtotal += itemCost;
            return acc;
        }, {});
    },

    /**
     * Renderizar grupo de tipo de APU
     */
    _renderApuTypeGroup(tableBody, apuType, groupData, aiuFactor) {
        const typeId = `type-${apuType.replace(/[^a-zA-Z0-9]/g, '-')}`;

        // Header del tipo
        const typeHeaderRow = document.createElement('tr');
        typeHeaderRow.className = 'bg-gray-200 hover:bg-gray-300 cursor-pointer transition-colors';
        typeHeaderRow.onclick = () => {
            document.querySelectorAll(`.${typeId}-child`).forEach(el => {
                el.classList.toggle('hidden');
            });
        };
        typeHeaderRow.innerHTML = `
            <td colspan="5" class="px-4 py-3 text-base font-bold text-gray-800 uppercase">
                ▶ ${Utils.escapeHtml(apuType)}
            </td>
            <td class="px-4 py-3 text-base font-bold text-gray-800 text-right">
                ${Utils.formatCurrency(groupData.subtotal)}
            </td>
            <td class="px-4 py-3 text-sm text-gray-500">-</td>
        `;
        tableBody.appendChild(typeHeaderRow);

        // Agrupar por descripción
        const descriptionGroups = this._groupByDescription(groupData.items);
        const sortedDescriptions = Object.keys(descriptionGroups).sort((a, b) => a.localeCompare(b));

        sortedDescriptions.forEach(groupName => {
            this._renderDescriptionGroup(tableBody, groupName, descriptionGroups[groupName], typeId, aiuFactor);
        });
    },

    /**
     * Agrupar por descripción
     */
    _groupByDescription(items) {
        return items.reduce((acc, item) => {
            const group = item.grupo || 'Ítems Varios';
            if (!acc[group]) acc[group] = [];
            acc[group].push(item);
            return acc;
        }, {});
    },

    /**
     * Renderizar grupo de descripción
     */
    _renderDescriptionGroup(tableBody, groupName, items, typeId, aiuFactor) {
        const groupId = `group-${groupName.replace(/[^a-zA-Z0-9]/g, '-')}`;

        const groupHeaderRow = document.createElement('tr');
        groupHeaderRow.className = `hidden ${typeId}-child bg-blue-100 hover:bg-blue-200 cursor-pointer transition-colors`;
        groupHeaderRow.onclick = (e) => {
            e.stopPropagation();
            document.querySelectorAll(`.${groupId}-item`).forEach(row => {
                row.classList.toggle('hidden');
            });
        };
        groupHeaderRow.innerHTML = `
            <td class="pl-8 pr-4 py-3 text-sm font-bold text-blue-800" colspan="6">
                ▶ ${Utils.escapeHtml(groupName)} (${items.length} ítems)
            </td>
            <td></td>
        `;
        tableBody.appendChild(groupHeaderRow);

        // Renderizar ítems
        items.forEach(item => {
            this._renderItemRow(tableBody, item, typeId, groupId, aiuFactor);
        });
    },

    /**
     * Renderizar fila de ítem
     */
    _renderItemRow(tableBody, item, typeId, groupId, aiuFactor) {
        const valorSuministroAIU = Utils.sanitizeNumber(item.VALOR_SUMINISTRO_UN, 0) * aiuFactor;
        const valorInstalacionAIU = Utils.sanitizeNumber(item.VALOR_INSTALACION_UN, 0) * aiuFactor;
        const valorConstruccionAIU = Utils.sanitizeNumber(item.VALOR_CONSTRUCCION_UN, 0) * aiuFactor;

        const tiempo = Utils.sanitizeNumber(item.TIEMPO_INSTALACION, null);
        const tiempoFormateado = tiempo !== null ? `${tiempo.toFixed(4)} Días/Un.` : 'N/A';

        const itemRow = document.createElement('tr');
        itemRow.className = `hidden ${typeId}-child ${groupId}-item hover:bg-gray-50 cursor-pointer transition-colors`;
        itemRow.onclick = (e) => {
            e.stopPropagation();
            ModalManager.open(item.CODIGO_APU);
        };

        const alertaIcon = item.alerta ?
            `<span class="text-yellow-500 ml-1" title="${Utils.escapeHtml(item.alerta)}"> ⚠️</span>` : '';

        itemRow.innerHTML = `
            <td class="pl-12 pr-4 py-4 text-sm font-medium text-gray-900">${Utils.escapeHtml(item.CODIGO_APU)}</td>
            <td class="px-4 py-4 text-sm text-gray-500">${Utils.escapeHtml(item.DESCRIPCION_APU)}${alertaIcon}</td>
            <td class="px-4 py-4 text-sm text-gray-500">${Utils.formatCurrency(valorSuministroAIU)}</td>
            <td class="px-4 py-4 text-sm text-gray-500">${Utils.formatCurrency(valorInstalacionAIU)}</td>
            <td class="px-4 py-4 text-sm font-semibold text-gray-700">${Utils.formatCurrency(valorConstruccionAIU)}</td>
            <td class="px-4 py-4 text-sm text-gray-500">${tiempoFormateado}</td>
        `;
        tableBody.appendChild(itemRow);
    },

    /**
     * Actualizar tabla del organizador
     */
    updateOrganizerTable() {
        const tableBody = document.getElementById('organizer-table-body');
        if (!tableBody) return;

        const budgetData = StateManager.getState('budgetData');
        tableBody.innerHTML = '';

        budgetData.forEach((item, index) => {
            const row = document.createElement('tr');
            row.className = 'hover:bg-gray-50 transition-colors';
            row.innerHTML = `
                <td class="px-4 py-4 text-sm text-gray-500">${Utils.escapeHtml(item.DESCRIPCION_APU)}</td>
                <td class="px-4 py-4">
                    <input type="text"
                           value="${Utils.escapeHtml(item.ZONA || '')}"
                           data-index="${index}"
                           data-field="ZONA"
                           class="organizer-input w-full p-1 border rounded-md text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                           placeholder="Ej: ZONA 1">
                </td>
                <td class="px-4 py-4">
                    <input type="text"
                           value="${Utils.escapeHtml(item.TIPO || '')}"
                           data-index="${index}"
                           data-field="TIPO"
                           class="organizer-input w-full p-1 border rounded-md text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                           placeholder="Ej: CUBIERTA">
                </td>
                <td class="px-4 py-4">
                    <input type="text"
                           value="${Utils.escapeHtml(item.MATERIAL || '')}"
                           data-index="${index}"
                           data-field="MATERIAL"
                           class="organizer-input w-full p-1 border rounded-md text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                           placeholder="Ej: TST">
                </td>
            `;
            tableBody.appendChild(row);
        });

        // Añadir event listeners con debounce
        document.querySelectorAll('.organizer-input').forEach(input => {
            input.addEventListener('input', Utils.debounce((e) => {
                const index = parseInt(e.target.dataset.index);
                const field = e.target.dataset.field;
                const value = e.target.value;
                this._updateItemData(index, field, value);
            }, CONFIG.DEBOUNCE_DELAY));
        });
    },

    /**
     * Actualizar datos de un ítem
     */
    _updateItemData(index, field, value) {
        const budgetData = StateManager.getState('budgetData');
        if (budgetData[index]) {
            budgetData[index][field] = value;
            StateManager.setState({ budgetData });
            SummaryController.updateSummaries();
            Logger.log('Dato actualizado', { index, field, value });
        }
    }
};

// ============================================================================
// CONTROLADOR DE RESÚMENES
// ============================================================================
const SummaryController = {
    /**
     * Actualizar todos los resúmenes
     */
    updateSummaries() {
        const budgetData = StateManager.getState('budgetData');
        const aiuPercentages = StateManager.getState('aiuPercentages');
        const aiuFactor = 1 + ((aiuPercentages.admin + aiuPercentages.imprev + aiuPercentages.util) / 100);

        const summaries = { ZONA: {}, TIPO: {}, MATERIAL: {} };

        budgetData.forEach(item => {
            const cost = Utils.sanitizeNumber(item.VALOR_CONSTRUCCION_UN, 0) *
                        Utils.sanitizeNumber(item.CANTIDAD_PRESUPUESTO, 0) * aiuFactor;

            ['ZONA', 'TIPO', 'MATERIAL'].forEach(key => {
                const group = item[key] || 'Sin Asignar';
                summaries[key][group] = (summaries[key][group] || 0) + cost;
            });
        });

        this._renderSummary('summary-zona', 'Resumen por Zona', summaries.ZONA);
        this._renderSummary('summary-tipo', 'Resumen por Tipo', summaries.TIPO);
        this._renderSummary('summary-material', 'Resumen por Material', summaries.MATERIAL);
    },

    /**
     * Renderizar un resumen específico
     */
    _renderSummary(elementId, title, data) {
        const element = document.getElementById(elementId);
        if (!element) return;

        let html = `<h3 class="text-lg font-semibold mb-3 text-gray-700">${title}</h3><div class="space-y-2">`;

        const entries = Object.entries(data).filter(([_, value]) => value > 0);

        if (entries.length === 0) {
            html += `<p class="text-sm text-gray-500">No hay datos para mostrar.</p>`;
        } else {
            entries.sort((a, b) => b[1] - a[1]).forEach(([key, value]) => {
                html += `
                    <div class="flex justify-between text-sm items-center p-2 hover:bg-gray-50 rounded transition-colors">
                        <span class="text-gray-600 font-medium">${Utils.escapeHtml(key)}</span>
                        <span class="font-semibold text-gray-800">${Utils.formatCurrency(value)}</span>
                    </div>
                `;
            });
        }

        element.innerHTML = html + '</div>';
    }
};

// ============================================================================
// CONTROLADOR DEL ESTIMADOR
// ============================================================================
const EstimatorController = {
    _isInitialized: false,

    /**
     * Inicializar estimador
     */
    initialize() {
        if (this._isInitialized) return;

        this._setupEventListeners();
        this._populateMaterialOptions();
        this._isInitialized = true;
        Logger.log('Estimador inicializado');
    },

    /**
     * Configurar event listeners
     */
    _setupEventListeners() {
        const productoSelect = document.getElementById('est-producto');
        if (productoSelect) {
            productoSelect.addEventListener('change', () => {
                this._populateMaterialOptions();
                this.updateEstimate();
            });
        }

        const estimatorInputs = [
            'est-material', 'est-cuadrilla', 'est-m2',
            'est-zona', 'est-izaje', 'est-seguridad'
        ];

        estimatorInputs.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => this.updateEstimate());
                if (element.type === 'number') {
                    element.addEventListener('input', Utils.debounce(() => this.updateEstimate(), CONFIG.DEBOUNCE_DELAY));
                }
            }
        });
    },

    /**
     * Poblar opciones de material según producto
     */
    _populateMaterialOptions() {
        const productoSelect = document.getElementById('est-producto');
        const materialSelect = document.getElementById('est-material');

        if (!productoSelect || !materialSelect) return;

        const producto = productoSelect.value;
        const materiales = CONFIG.MATERIAL_MAP[producto] || [];

        materialSelect.innerHTML = '';
        materiales.forEach(material => {
            const option = document.createElement('option');
            option.value = material;
            option.textContent = material;
            materialSelect.appendChild(option);
        });
    },

    /**
     * Actualizar estimación
     */
    async updateEstimate() {
        const rawData = StateManager.getState('rawData');
        if (!rawData || Object.keys(rawData).length === 0) {
            Logger.warn('Intento de actualizar estimación sin datos cargados');
            return;
        }

        const params = this._getEstimatorParams();

        // Validar parámetros
        if (!this._validateParams(params)) {
            return;
        }

        const logContainer = document.getElementById('estimator-log-container');
        const logEl = document.getElementById('estimator-log');

        if (logContainer) logContainer.classList.remove('hidden');
        if (logEl) logEl.textContent = 'Calculando...';

        try {
            const response = await fetch(CONFIG.API_ENDPOINTS.ESTIMATE, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(params),
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error(`Error ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (logEl) {
                logEl.textContent = result.log || 'No se recibió log del servidor.';
            }

            if (result.error) {
                throw new Error(result.error);
            }

            this._renderResults(result, params.m2);
            Logger.log('Estimación actualizada', { params, result });

        } catch (error) {
            Logger.error('Error al actualizar estimación', error);
            if (logEl) {
                logEl.textContent = `Error: ${error.message}`;
            }
            UIManager.showStatus(`Error en estimación: ${error.message}`, 'error');
        }
    },

    /**
     * Obtener parámetros del estimador
     */
    _getEstimatorParams() {
        return {
            producto: document.getElementById('est-producto')?.value || 'CUBIERTA',
            material: document.getElementById('est-material')?.value || '',
            cuadrilla: document.getElementById('est-cuadrilla')?.value || '5',
            m2: Utils.sanitizeNumber(document.getElementById('est-m2')?.value, 1, 0.01),
            zona: document.getElementById('est-zona')?.value || 'ZONA 0',
            izaje: document.getElementById('est-izaje')?.value || 'MANUAL',
            seguridad: document.getElementById('est-seguridad')?.value || 'NORMAL'
        };
    },

    /**
     * Validar parámetros
     */
    _validateParams(params) {
        if (!params.material) {
            UIManager.showStatus('Seleccione un material', 'warning');
            return false;
        }

        if (params.m2 <= 0) {
            UIManager.showStatus('La cantidad debe ser mayor a 0', 'warning');
            UIManager.shakeElement('est-m2');
            return false;
        }

        return true;
    },

    /**
     * Renderizar resultados
     */
    _renderResults(result, m2) {
        const aiuPercentages = StateManager.getState('aiuPercentages');
        const aiuFactor = 1 + ((aiuPercentages.admin + aiuPercentages.imprev + aiuPercentages.util) / 100);

        const valorSuministro = Utils.sanitizeNumber(result.valor_suministro, 0);
        const valorInstalacion = Utils.sanitizeNumber(result.valor_instalacion, 0);
        const tiempoInstalacion = Utils.sanitizeNumber(result.tiempo_instalacion, 0);

        const valorSuministroAIU = valorSuministro * aiuFactor;
        const valorInstalacionAIU = valorInstalacion * aiuFactor;
        const valorConstruccionAIU = (valorSuministro + valorInstalacion) * aiuFactor;

        const elements = {
            'est-res-suministro': Utils.formatCurrency(valorSuministroAIU),
            'est-res-instalacion': Utils.formatCurrency(valorInstalacionAIU),
            'est-res-construccion': Utils.formatCurrency(valorConstruccionAIU),
            'est-res-tiempo': tiempoInstalacion.toFixed(4)
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });
    }
};

// ============================================================================
// CONTROLADOR PRINCIPAL DE LA APLICACIÓN
// ============================================================================
const AppController = {
    /**
     * Procesar archivos cargados
     */
    async procesarArchivos() {
        const presupuestoFile = document.getElementById('presupuesto_file')?.files[0];
        const apusFile = document.getElementById('apus_file')?.files[0];
        const insumosFile = document.getElementById('insumos_file')?.files[0];

        try {
            // Validar archivos
            Utils.validateFile(presupuestoFile);
            Utils.validateFile(apusFile);
            Utils.validateFile(insumosFile);

            UIManager.updateFileStatus('presupuesto', 'success', 'Archivo válido');
            UIManager.updateFileStatus('apus', 'success', 'Archivo válido');
            UIManager.updateFileStatus('insumos', 'success', 'Archivo válido');

        } catch (error) {
            UIManager.showStatus(error.message, 'error');
            Logger.error('Validación de archivos falló', error);
            return;
        }

        const formData = new FormData();
        formData.append('presupuesto', presupuestoFile);
        formData.append('apus', apusFile);
        formData.append('insumos', insumosFile);

        StateManager.setState({ isProcessing: true });
        UIManager.toggleLoader(true);
        UIManager.toggleProcessButton(false);
        UIManager.showStatus("Procesando archivos...", 'info');

        try {
            Logger.log("Iniciando procesamiento de archivos");

            const response = await fetch(CONFIG.API_ENDPOINTS.UPLOAD, {
                method: 'POST',
                body: formData,
                credentials: 'include'
            });

            Logger.log(`Respuesta del servidor: ${response.status}`);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Error del servidor (${response.status}): ${errorText}`);
            }

            const rawData = await response.json();
            Logger.log("Datos JSON recibidos correctamente");

            if (rawData.error) {
                throw new Error(rawData.error);
            }

            // Validar estructura de datos
            Utils.validateDataStructure(rawData, ['presupuesto']);

            const budgetData = Utils.deepClone(rawData.presupuesto);

            StateManager.setState({ rawData, budgetData });

            UIManager.toggleMainContent(true);
            UIManager.showStatus("¡Análisis completado! Ya puedes simular costos y organizar tu proyecto.", 'success');

            this._setupCalculationListeners();
            EstimatorController.initialize();
            this.updateAllCalculations();

            Logger.log("Procesamiento completado exitosamente");

        } catch (error) {
            Logger.error("Error en procesamiento", error);
            UIManager.showStatus(`Error en el procesamiento: ${error.message}`, 'error');
            UIManager.shakeElement('upload-container');
        } finally {
            StateManager.setState({ isProcessing: false });
            UIManager.toggleLoader(false);
            UIManager.toggleProcessButton(true);
        }
    },

    /**
     * Configurar listeners para cálculos
     */
    _setupCalculationListeners() {
        const aiuInputs = ['admin_percent', 'imprev_percent', 'util_percent'];

        aiuInputs.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('input', Utils.debounce((e) => {
                    const value = Utils.sanitizeNumber(
                        e.target.value,
                        CONFIG.AIU_DEFAULTS[id.replace('_percent', '')],
                        CONFIG.AIU_LIMITS.min,
                        CONFIG.AIU_LIMITS.max
                    );

                    e.target.value = value;

                    const aiuPercentages = StateManager.getState('aiuPercentages');
                    aiuPercentages[id.replace('_percent', '')] = value;
                    StateManager.setState({ aiuPercentages });

                    this.updateAllCalculations();
                }, CONFIG.DEBOUNCE_DELAY));
            }
        });
    },

    /**
     * Actualizar todos los cálculos
     */
    updateAllCalculations() {
        try {
            TableController.updateSimulatorTable();
            TableController.updateOrganizerTable();
            SummaryController.updateSummaries();

            if (EstimatorController._isInitialized) {
                EstimatorController.updateEstimate();
            }
        } catch (error) {
            Logger.error('Error al actualizar cálculos', error);
            UIManager.showStatus('Error al actualizar visualización', 'error');
        }
    },

    /**
     * Cambiar pestaña activa
     */
    switchTab(tabName) {
        const validTabs = ['simulador', 'organizador', 'estimador'];

        if (!validTabs.includes(tabName)) {
            Logger.warn(`Pestaña inválida: ${tabName}`);
            return;
        }

        // Ocultar todo el contenido
        document.querySelectorAll('.tab-content').forEach(el => {
            el.classList.add('hidden');
            el.setAttribute('aria-hidden', 'true');
        });

        // Mostrar contenido seleccionado
        const contentEl = document.getElementById(`content-${tabName}`);
        if (contentEl) {
            contentEl.classList.remove('hidden');
            contentEl.setAttribute('aria-hidden', 'false');
        }

        // Actualizar tabs
        document.querySelectorAll('nav[role="tablist"] button').forEach(el => {
            el.classList.remove('tab-active');
            el.classList.add('tab-inactive');
            el.setAttribute('aria-selected', 'false');
        });

        const activeTab = document.getElementById(`tab-${tabName}`);
        if (activeTab) {
            activeTab.classList.add('tab-active');
            activeTab.classList.remove('tab-inactive');
            activeTab.setAttribute('aria-selected', 'true');
        }

        StateManager.setState({ currentTab: tabName });
        Logger.log(`Pestaña cambiada a: ${tabName}`);
    }
};

// ============================================================================
// INICIALIZACIÓN
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    Logger.log('Aplicación inicializada');

    // Validar elementos críticos del DOM
    const criticalElements = [
        'upload-container', 'main-content', 'loader', 'status',
        'presupuesto_file', 'apus_file', 'insumos_file', 'process-btn'
    ];

    const missingElements = criticalElements.filter(id => !document.getElementById(id));
    if (missingElements.length > 0) {
        Logger.error('Elementos críticos faltantes en el DOM', missingElements);
        UIManager.showStatus('Error de inicialización. Recargue la página.', 'error');
    }

    // Suscribirse a cambios de estado para logging
    StateManager.subscribe((newState, prevState) => {
        if (newState.isProcessing !== prevState.isProcessing) {
            Logger.log(`Estado de procesamiento: ${newState.isProcessing}`);
        }
    });

    Logger.log('Sistema listo para uso');
});

// Cerrar modal al hacer clic fuera
document.addEventListener('click', (e) => {
    const modal = document.getElementById('apu-modal');
    if (modal && e.target === modal) {
        ModalManager.close();
    }
});

// Manejo global de errores
window.addEventListener('error', (event) => {
    Logger.error('Error global capturado', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    Logger.error('Promesa rechazada no manejada', event.reason);
});