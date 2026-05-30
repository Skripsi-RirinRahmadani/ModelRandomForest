// URL Konfigurasi FastAPI
const API_URL = "http://localhost:8000";
let apiIsOnline = false;

// Element DOM
const form = document.getElementById("predict-form");
const predictBtn = document.getElementById("predict-btn");
const simulateBtn = document.getElementById("simulate-btn");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const offlineAlert = document.getElementById("offline-alert");
const placeholder = document.getElementById("results-placeholder");
const resultsActual = document.getElementById("results-actual");

const resKecamatan = document.getElementById("result-kecamatan");
const resConfidence = document.getElementById("result-confidence");
const recsGrid = document.getElementById("recs-grid");

// Slider DOM dan Live Value Updates
const sliders = [
    { id: "ph_tanah", valId: "ph-val", suffix: "", decimals: 2 },
    { id: "suhu_c", valId: "suhu-val", suffix: "°C", decimals: 1 },
    { id: "curah_hujan_mm", valId: "hujan-val", suffix: " mm", decimals: 0 },
    { id: "elevasi_mdpl", valId: "elevasi-val", suffix: " mdpl", decimals: 0 },
    { id: "intensitas_matahari_jam", valId: "matahari-val", suffix: " jam", decimals: 1 }
];

sliders.forEach(slider => {
    const el = document.getElementById(slider.id);
    const valEl = document.getElementById(slider.valId);
    
    el.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        valEl.textContent = val.toFixed(slider.decimals) + slider.suffix;
    });
});

// Periksa Status API saat Startup
async function checkApiStatus() {
    try {
        const res = await fetch(API_URL + "/", { method: "GET", signal: AbortSignal.timeout(1500) });
        if (res.ok) {
            apiIsOnline = true;
            statusDot.className = "status-dot online";
            statusText.textContent = "API Server Online";
            offlineAlert.style.display = "none";
        } else {
            setApiOffline();
        }
    } catch (e) {
        setApiOffline();
    }
}

function setApiOffline() {
    apiIsOnline = false;
    statusDot.className = "status-dot";
    statusText.textContent = "API Offline (Simulasi Mode)";
}

// Panggil Pengecekan API di awal
checkApiStatus();
setInterval(checkApiStatus, 10000); // Cek ulang setiap 10 detik

// Form Submit Handler
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await runAnalysis(false);
});

simulateBtn.addEventListener("click", async () => {
    await runAnalysis(true);
});

async function runAnalysis(forceSimulation = false) {
    // Set Loading State
    predictBtn.disabled = true;
    predictBtn.innerHTML = `<span class="spinner"></span> Menganalisis...`;
    
    // Ambil Data Input
    const ph = parseFloat(document.getElementById("ph_tanah").value);
    const suhu = parseFloat(document.getElementById("suhu_c").value);
    const hujan = parseFloat(document.getElementById("curah_hujan_mm").value);
    const elevasi = parseFloat(document.getElementById("elevasi_mdpl").value);
    const air = document.getElementById("ketersediaan_air").value;
    const matahari = parseFloat(document.getElementById("intensitas_matahari_jam").value);
    
    const payload = {
        ph_tanah: ph,
        suhu_c: suhu,
        curah_hujan_mm: hujan,
        elevasi_mdpl: elevasi,
        ketersediaan_air: air,
        intensitas_matahari_jam: matahari
    };

    // Simulasi delay kosmetik agar UI terasa premium
    await new Promise(r => setTimeout(r, 600));

    if (apiIsOnline && !forceSimulation) {
        try {
            const res = await fetch(API_URL + "/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            
            if (!res.ok) throw new Error("Gagal mengambil prediksi dari server.");
            
            const data = await res.json();
            if (data.status === "success") {
                displayResults(data.identified_location, data.location_confidence, data.recommendations);
                offlineAlert.style.display = "none";
            } else {
                throw new Error(data.message || "Unknown API Error");
            }
        } catch (err) {
            console.warn("API Error, falling back to Simulation Mode:", err);
            runSimulationFallback(payload);
        }
    } else {
        runSimulationFallback(payload);
    }

    // Reset Loading State
    predictBtn.disabled = false;
    predictBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Analisis Lingkungan`;
}

// Menampilkan Hasil ke DOM
function displayResults(kecamatan, confidence, recommendations) {
    placeholder.style.display = "none";
    resultsActual.style.display = "block";
    
    resKecamatan.textContent = kecamatan;
    resKecamatan.style.color = "#38bdf8";
    resConfidence.textContent = confidence;
    
    // Bersihkan Grid
    recsGrid.innerHTML = "";
    
    // Icon peta tanaman untuk visualisasi
    const plantIcons = {
        "Bayam": "fa-leaf",
        "Cabe Besar": "fa-pepper-hot",
        "Cabe Keriting": "fa-pepper-hot",
        "Cabe Rawit": "fa-pepper-hot",
        "Kacang Panjang": "fa-seedling",
        "Kangkung": "fa-envira",
        "Ketimun": "fa-cubes",
        "Semangka": "fa-solid fa-cookie-bite",
        "Terung": "fa-egg",
        "Tomat": "fa-apple-whole"
    };

    recommendations.forEach(rec => {
        const icon = plantIcons[rec.tanaman] || "fa-seedling";
        const scorePct = parseFloat(rec.kecocokan);
        
        // Buat Kartu Rekomendasi
        const card = document.createElement("div");
        card.className = "rec-card";
        card.innerHTML = `
            <div class="rec-header">
                <span class="rec-plant">${rec.tanaman}</span>
                <i class="fa-solid ${icon} rec-icon"></i>
            </div>
            <div class="rec-variety-box">
                <div class="rec-variety-label">Varietas Terbaik</div>
                <div class="rec-variety">${rec.varietas}</div>
            </div>
            <div class="score-bar-wrapper">
                <div class="score-label">
                    <span>Tingkat Kecocokan</span>
                    <span class="score-val">${rec.kecocokan}</span>
                </div>
                <div class="score-bg">
                    <div class="score-fill" id="fill-${rec.tanaman.replace(/\s+/g, '-')}"></div>
                </div>
            </div>
        `;
        recsGrid.appendChild(card);
        
        // Animasi Progress Bar masuk secara perlahan
        setTimeout(() => {
            const fill = document.getElementById(`fill-${rec.tanaman.replace(/\s+/g, '-')}`);
            if (fill) fill.style.width = `${scorePct}%`;
        }, 100);
    });
}

// Simulasi Klien Sisi Lokal (Fallback Cerdas jika Server API Mati)
function runSimulationFallback(inputs) {
    offlineAlert.style.display = "flex";
    
    // 1. Logika Determinasi Geografis (Meniru Sidik Jari Kecamatan di Aceh Utara)
    let predictedKecamatan = "Lhoksukon";
    let baseConfidence = 96.2;

    if (inputs.elevasi_mdpl > 400) {
        predictedKecamatan = inputs.ph_tanah < 5.5 ? "Geureudong Pase" : "Nisam Antara";
    } else if (inputs.elevasi_mdpl > 200) {
        predictedKecamatan = inputs.suhu_c < 25.5 ? "Sawang" : "Cot Girek";
    } else if (inputs.elevasi_mdpl < 15) {
        predictedKecamatan = inputs.ph_tanah > 6.0 ? "Lapang" : "Seunuddon";
    } else {
        // Dataran sedang, pengelompokan berdasarkan pH & Suhu
        if (inputs.ph_tanah > 6.0) predictedKecamatan = "Muara Batu";
        else if (inputs.suhu_c > 26.8) predictedKecamatan = "Baktiya";
        else if (inputs.curah_hujan_mm > 2200) predictedKecamatan = "Langkahan";
        else predictedKecamatan = "Dewantara";
    }

    // Variasi acak kecil pada tingkat keyakinan (untuk estetika dinamis)
    const confidenceFormatted = (baseConfidence + (Math.sin(inputs.ph_tanah * 10) * 2)).toFixed(2) + "%";

    // 2. Daftar Tanaman & Varietas Referensi
    const cropVarieties = [
        { tanaman: "Bayam", varietas: "Bangkok" },
        { tanaman: "Cabe Besar", varietas: "Gada F1" },
        { tanaman: "Cabe Keriting", varietas: "Kencana" },
        { tanaman: "Cabe Rawit", varietas: "Bara" },
        { tanaman: "Kacang Panjang", varietas: "Parade" },
        { tanaman: "Kangkung", varietas: "Bina" },
        { tanaman: "Ketimun", varietas: "Hercules F1" },
        { tanaman: "Semangka", varietas: "Inden F1" },
        { tanaman: "Terung", varietas: "Mustang F1" },
        { tanaman: "Tomat", varietas: "Tymoti F1" }
    ];

    // 3. Kalkulasi Tingkat Kecocokan Simulasi berbasis Kemiripan Kondisi Ideal
    const recommendations = cropVarieties.map(crop => {
        // Tetapkan profil ideal acak untuk setiap tanaman agar tingkat kecocokan dinamis terhadap input
        let idealPH = 6.0;
        let idealSuhu = 26.0;
        let idealHujan = 1800;
        
        if (crop.tanaman.includes("Cabe")) {
            idealPH = 6.2; idealSuhu = 27.5; idealHujan = 2100;
        } else if (crop.tanaman === "Tomat") {
            idealPH = 6.4; idealSuhu = 24.5; idealHujan = 1500;
        } else if (crop.tanaman === "Kangkung" || crop.tanaman === "Bayam") {
            idealPH = 5.8; idealSuhu = 28.0; idealHujan = 2500;
        }

        // Kalkulasi persentase kecocokan lingkungan menggunakan jarak absolut ternormalisasi
        const distPH = Math.abs(inputs.ph_tanah - idealPH) / 3; // Skala variasi pH ~3
        const distSuhu = Math.abs(inputs.suhu_c - idealSuhu) / 15; // Skala variasi suhu ~15
        const distHujan = Math.abs(inputs.curah_hujan_mm - idealHujan) / 2500; // Skala hujan ~2500
        
        const avgDistance = (distPH + distSuhu + distHujan) / 3;
        let matchScore = (1 - avgDistance) * 100;
        
        // Batasi nilai agar realistis
        if (matchScore > 98) matchScore = 98 - (Math.random() * 2);
        if (matchScore < 45) matchScore = 45 + (Math.random() * 10);

        return {
            tanaman: crop.tanaman,
            varietas: crop.varietas,
            kecocokan: matchScore.toFixed(2) + "%"
        };
    });

    // Tampilkan data simulasi lokal
    displayResults(predictedKecamatan, confidenceFormatted, recommendations);
}
