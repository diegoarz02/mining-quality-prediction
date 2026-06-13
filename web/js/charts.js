let shapChartInstance = null;
let simChartInstance = null;

export function renderShapChart(canvasId, contributions) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const labels = contributions.map(c => c.feature);
    const data = contributions.map(c => c.shap_value);
    const colors = data.map(v => v > 0 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.7)');
    
    if (shapChartInstance) {
        shapChartInstance.destroy();
    }
    
    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Impacto en Sílice',
                data: data,
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Contribución SHAP' }
                }
            }
        }
    });
}

export function renderSimChart(canvasId, baseTrajectory, simTrajectory) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const labels = Array.from({length: baseTrajectory.length}, (_, i) => `H+${i+1}`);
    
    if (simChartInstance) {
        simChartInstance.destroy();
    }
    
    simChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Base (Sin cambios)',
                    data: baseTrajectory,
                    borderColor: '#94a3b8',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'Escenario',
                    data: simTrajectory,
                    borderColor: '#1B3D6E',
                    backgroundColor: 'rgba(27, 61, 110, 0.1)',
                    fill: true,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: { display: true, text: '% Sílice Predicha' }
                }
            }
        }
    });
}
