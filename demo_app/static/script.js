let signalChart = null;

document.addEventListener('DOMContentLoaded', () => {
    loadSamples();
});

async function loadSamples() {
    try {
        const response = await fetch(`/api/samples?t=${Date.now()}`);
        const samples = await response.json();

        const list = document.getElementById('sample-list');
        list.innerHTML = '';

        samples.forEach(sample => {
            const item = document.createElement('div');
            item.className = 'sample-item';
            item.innerHTML = `
                <span class="sample-id">Sample #${sample.id}</span>
                <span class="sample-label">${sample.label_name}</span>
            `;
            item.onclick = () => selectSample(sample.id, item);
            list.appendChild(item);
        });
    } catch (error) {
        console.error('Error loading samples:', error);
        document.getElementById('sample-list').innerHTML = '<p class="error">Failed to load samples</p>';
    }
}

async function selectSample(id, element) {
    // Update UI
    document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('active'));
    element.classList.add('active');

    // Show loading state
    document.getElementById('prediction-result').innerHTML = '<span class="placeholder-text">Analyzing...</span>';

    try {
        const response = await fetch(`/api/predict/${id}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Unknown server error');
        }
        const data = await response.json();

        updatePredictionUI(data);
        updateChart(data.signals);
    } catch (error) {
        console.error('Error predicting:', error);
        document.getElementById('prediction-result').innerHTML = `<span class="error-text">Error: ${error.message}</span>`;
    }
}

function updatePredictionUI(data) {
    const predEl = document.getElementById('prediction-result');
    const confBar = document.getElementById('confidence-bar');
    const confText = document.getElementById('confidence-text');

    const className = `status-${data.prediction_name.toLowerCase()}`;
    predEl.innerHTML = `<span class="${className}">${data.prediction_name}</span>`;
    predEl.style.opacity = 0;
    setTimeout(() => predEl.style.opacity = 1, 50);

    const confidence = (data.probabilities[data.prediction] * 100).toFixed(1);
    confBar.style.width = `${confidence}%`;
    confText.innerText = `Confidence: ${confidence}%`;
}

function updateChart(signals) {
    const ctx = document.getElementById('signalChart').getContext('2d');

    if (signalChart) {
        signalChart.destroy();
    }

    const labels = Array.from({ length: signals.eda.length }, (_, i) => i);

    signalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'EDA (Phasic/Tonic)',
                    data: signals.eda,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y'
                },
                {
                    label: 'ACC Magnitude',
                    data: signals.acc,
                    borderColor: '#ec4899',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8', font: { family: 'Outfit' } }
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}
