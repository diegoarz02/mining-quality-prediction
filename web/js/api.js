export async function fetchHealth() {
    try {
        const res = await fetch('/health');
        return await res.json();
    } catch (e) {
        console.error("Error fetching health:", e);
        return null;
    }
}

export async function fetchReference() {
    const res = await fetch('/reference');
    if (!res.ok) throw new Error("Error fetching reference");
    return await res.json();
}

export async function fetchPrediction(features) {
    const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
    });
    if (!res.ok) throw new Error("Error fetching prediction");
    return await res.json();
}

export async function fetchExplain(features) {
    const res = await fetch('/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
    });
    if (!res.ok) throw new Error("Error fetching explain");
    return await res.json();
}

export async function fetchSimulateSustained(base_features, deltas) {
    const res = await fetch('/simulate-sustained', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base_features, deltas })
    });
    if (!res.ok) throw new Error("Error fetching simulate sustained");
    return await res.json();
}

export async function fetchReport(features) {
    const res = await fetch('/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
    });
    if (!res.ok) throw new Error("Error fetching report");
    return await res.json();
}
