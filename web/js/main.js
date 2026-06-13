import { 
    fetchHealth, fetchReference, fetchPrediction, 
    fetchExplain, fetchSimulateSustained, fetchReport 
} from './api.js';

import { 
    updateBadges, renderPredictionForm, renderSimulatorSliders, 
    updatePredictionDisplay, renderReport 
} from './ui.js';

import { renderShapChart, renderSimChart } from './charts.js';

let currentReference = null;
let currentBaseFeatures = {};

async function init() {
    // 1. Health check
    const health = await fetchHealth();
    updateBadges(health);
    
    if (!health || health.status !== 'ok') return;
    
    // 2. Fetch reference data
    currentReference = await fetchReference();
    renderPredictionForm(currentReference);
    renderSimulatorSliders(currentReference);
    
    // 3. Initial Prediction
    await handlePredict();
    
    // 4. Initial Simulator update
    await handleSimulate();
    
    // Listeners
    document.getElementById('btn-predict').addEventListener('click', handlePredict);
    document.getElementById('threshold-input').addEventListener('change', () => {
        // Just re-evaluate prediction UI without fetching
        handlePredict(); 
    });
    
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(s => s.addEventListener('change', handleSimulate));
    
    document.getElementById('btn-reset-sim').addEventListener('click', () => {
        sliders.forEach(s => {
            s.value = s.dataset.base;
            document.getElementById(`val-${s.dataset.key}`).textContent = parseFloat(s.dataset.base).toFixed(2);
        });
        handleSimulate();
    });
    
    document.getElementById('btn-report').addEventListener('click', handleReport);
}

function getFormFeatures() {
    const features = {};
    const inputs = document.querySelectorAll('#prediction-form input');
    inputs.forEach(input => {
        features[input.name] = parseFloat(input.value);
    });
    return features;
}

async function handlePredict() {
    const features = getFormFeatures();
    currentBaseFeatures = features;
    
    const threshold = parseFloat(document.getElementById('threshold-input').value);
    
    // Predict
    const predRes = await fetchPrediction(features);
    updatePredictionDisplay(predRes, threshold);
    
    // Explain
    const explainRes = await fetchExplain(features);
    renderShapChart('shap-chart', explainRes.contributions);
    
    // Update simulator base values
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(s => {
        const key = s.dataset.key;
        if (features[key] !== undefined) {
            s.dataset.base = features[key];
            // Si el slider no ha sido movido del base, actualizarlo visualmente
            s.value = features[key];
            document.getElementById(`val-${key}`).textContent = features[key].toFixed(2);
        }
    });
    
    await handleSimulate();
}

async function handleSimulate() {
    const deltas = {};
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(s => {
        const key = s.dataset.key;
        const val = parseFloat(s.value);
        const base = parseFloat(s.dataset.base);
        if (val !== base) {
            deltas[key] = val - base;
        }
    });
    
    const res = await fetchSimulateSustained(currentBaseFeatures, deltas);
    renderSimChart('sim-chart', res.trajectory_base, res.trajectory);
}

async function handleReport() {
    const features = getFormFeatures();
    
    // Apply simulator deltas to features for report
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(s => {
        const key = s.dataset.key;
        const val = parseFloat(s.value);
        features[key] = val; // Replace with simulated value
    });
    
    const container = document.getElementById('report-content');
    const loading = document.getElementById('report-loading');
    const meta = document.getElementById('report-meta');
    
    container.classList.add('hidden');
    meta.classList.add('hidden');
    loading.classList.remove('hidden');
    
    const res = await fetchReport(features);
    renderReport(res);
}

// Start
document.addEventListener('DOMContentLoaded', init);
