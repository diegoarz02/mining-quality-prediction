export const TEST_MAE = 0.4599;

export function updateBadges(health) {
    const statusBadge = document.getElementById('status-badge');
    const sourceBadge = document.getElementById('model-source-badge');
    
    if (health && health.status === 'ok') {
        statusBadge.textContent = 'Servicio OK';
        statusBadge.className = 'badge ok';
        sourceBadge.textContent = `Modelo: ${health.model_source}`;
        sourceBadge.className = 'badge ok';
    } else {
        statusBadge.textContent = 'Servicio Caído';
        statusBadge.className = 'badge error';
        sourceBadge.textContent = 'Error';
        sourceBadge.className = 'badge error';
    }
}

export function renderPredictionForm(reference) {
    const form = document.getElementById('prediction-form');
    form.innerHTML = '';
    
    // Agrupar por sentido común de planta
    const order = [
        '% Silica Concentrate__lag1h',
        '% Silica Feed',
        '% Iron Feed',
        'Ore Pulp Flow__mean',
        'Ore Pulp Density__mean',
        'Ore Pulp pH__mean',
        'Amina Flow__mean',
        'Starch Flow__mean'
    ];
    
    // Add known inputs
    for (const key of order) {
        if (reference.reference[key] !== undefined) {
            createFormGroup(form, key, reference.reference[key]);
        }
    }
}

function createFormGroup(container, key, value) {
    const div = document.createElement('div');
    div.className = 'form-group';
    
    const label = document.createElement('label');
    label.textContent = formatFeatureName(key);
    label.htmlFor = `input-${key}`;
    
    const input = document.createElement('input');
    input.type = 'number';
    input.id = `input-${key}`;
    input.name = key;
    input.step = 'any';
    input.value = value.toFixed(4);
    
    div.appendChild(label);
    div.appendChild(input);
    container.appendChild(div);
}

export function renderSimulatorSliders(reference) {
    const container = document.getElementById('simulator-sliders');
    container.innerHTML = '';
    
    const manipulable = [
        'Amina Flow__mean',
        'Starch Flow__mean',
        'Ore Pulp pH__mean',
        'Flotation Column 01 Air Flow__mean',
        'Flotation Column 01 Level__mean'
    ];
    
    for (const key of manipulable) {
        if (reference.p5[key] !== undefined) {
            const min = reference.p5[key];
            const max = reference.p95[key];
            const val = reference.reference[key];
            createSliderGroup(container, key, min, max, val);
        }
    }
}

function createSliderGroup(container, key, min, max, value) {
    const div = document.createElement('div');
    div.className = 'simulator-slider';
    
    const labelContainer = document.createElement('label');
    const nameSpan = document.createElement('span');
    nameSpan.textContent = formatFeatureName(key);
    
    const valSpan = document.createElement('span');
    valSpan.id = `val-${key}`;
    valSpan.textContent = value.toFixed(2);
    valSpan.style.color = 'var(--color-primary)';
    
    labelContainer.appendChild(nameSpan);
    labelContainer.appendChild(valSpan);
    
    const input = document.createElement('input');
    input.type = 'range';
    input.id = `slider-${key}`;
    input.dataset.key = key;
    input.min = min;
    input.max = max;
    input.step = (max - min) / 100;
    input.value = value;
    input.dataset.base = value; // Guardar valor base
    
    input.addEventListener('input', (e) => {
        document.getElementById(`val-${key}`).textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    div.appendChild(labelContainer);
    div.appendChild(input);
    container.appendChild(div);
}

export function updatePredictionDisplay(predictionResponse, threshold) {
    const val = predictionResponse.predicted_silica;
    document.getElementById('pred-silica').textContent = `${val.toFixed(2)}%`;
    
    // Traffic light
    const isExceed = val > threshold;
    const isWarn = val > (threshold - TEST_MAE);
    
    document.querySelectorAll('.light').forEach(l => l.classList.remove('active'));
    const tlMsg = document.getElementById('tl-message');
    
    if (isExceed) {
        document.getElementById('tl-red').classList.add('active');
        tlMsg.textContent = 'Alerta: Sílice fuera de especificación';
        tlMsg.style.color = 'var(--color-red)';
    } else if (isWarn) {
        document.getElementById('tl-amber').classList.add('active');
        tlMsg.textContent = 'Precaución: Cerca del umbral';
        tlMsg.style.color = 'var(--color-amber)';
    } else {
        document.getElementById('tl-green').classList.add('active');
        tlMsg.textContent = 'Dentro de especificación';
        tlMsg.style.color = 'var(--color-green)';
    }
}

export function renderReport(reportResponse) {
    const container = document.getElementById('report-content');
    const loading = document.getElementById('report-loading');
    const meta = document.getElementById('report-meta');
    const badge = document.getElementById('report-source-badge');
    
    loading.classList.add('hidden');
    container.classList.remove('hidden');
    
    let html = '';
    
    if (reportResponse.report.error) {
         html += `<p class="warning-text">${reportResponse.report.error}</p>`;
    }
    
    const r = reportResponse.report;
    if (r.situation) {
        html += `<h3>1. Situación Actual</h3><p>${r.situation}</p>`;
    }
    if (r.drivers) {
        html += `<h3>2. Principales Drivers</h3><ul>`;
        r.drivers.forEach(d => html += `<li>${d}</li>`);
        html += `</ul>`;
    }
    if (r.scenarios) {
        html += `<h3>3. Escenarios Evaluados</h3><ul>`;
        r.scenarios.forEach(s => html += `<li>${s}</li>`);
        html += `</ul>`;
    }
    if (r.recommendation) {
        html += `<h3>4. Recomendación Operativa</h3><p><strong>Impacto estimado: ${r.recommendation.impact}</strong></p><p>${r.recommendation.action}</p>`;
    }
    if (r.risks_and_limits) {
        html += `<h3>5. Riesgos y Límites</h3><ul>`;
        r.risks_and_limits.forEach(rsk => html += `<li>${rsk}</li>`);
        html += `</ul>`;
    }
    
    if (!html) {
        html = `<p>Error procesando el informe.</p><pre>${JSON.stringify(r, null, 2)}</pre>`;
    }
    
    container.innerHTML = html;
    
    meta.classList.remove('hidden');
    if (reportResponse.is_fallback) {
        badge.textContent = 'Generado por reglas (Fallback)';
        badge.className = 'badge error';
    } else {
        badge.textContent = 'Generado por LLM (LangGraph)';
        badge.className = 'badge ok';
    }
}

function formatFeatureName(key) {
    let name = key.replace('__mean', '').replace('__lag1h', ' (1h atrás)');
    return name;
}
