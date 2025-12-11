// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const captionInput = document.getElementById('captionInput');
const auditBtn = document.getElementById('auditBtn');

// Image Upload Logic
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            preview.classList.add('visible');
            dropZone.querySelector('p').classList.add('hidden');
        }
        reader.readAsDataURL(file);
    }
});

// Audit Logic
auditBtn.addEventListener('click', async () => {
    // 1. Validate
    if (!fileInput.files[0] || !captionInput.value) {
        alert("Please upload an image and write a caption!");
        return;
    }

    auditBtn.innerText = "Analyzing...";
    auditBtn.disabled = true;

    try {
        // 2. Call Image API
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const imgRes = await fetch('/predict-image', {
            method: 'POST',
            body: formData
        });
        const imgData = await imgRes.json();

        // 3. Call Text API
        const txtRes = await fetch('/predict-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ caption: captionInput.value })
        });
        const txtData = await txtRes.json();

        // 4. Update UI
        updateUI(imgData, txtData);

    } catch (error) {
        console.error(error);
        alert("Error connecting to server.");
    }

    auditBtn.innerText = "🚀 Run Audit";
    auditBtn.disabled = false;
});

function updateUI(imgData, txtData) {
    // Update Image Score
    const score = imgData.score;
    const bar = document.getElementById('imgBar');
    document.getElementById('imgScore').innerText = score + " / 100";
    bar.style.width = score + "%";
    
    if(score > 70) bar.style.backgroundColor = "var(--success)";
    else if(score > 40) bar.style.backgroundColor = "var(--warning)";
    else bar.style.backgroundColor = "var(--danger)";

    // Update Text Result
    const badge = document.getElementById('txtBadge');
    badge.innerText = txtData.label;
    document.getElementById('txtConf').innerText = `Confidence: ${txtData.confidence}%`;

    if(txtData.label.includes("High")) badge.style.backgroundColor = "var(--success)";
    else if(txtData.label.includes("Average")) badge.style.backgroundColor = "var(--warning)";
    else badge.style.backgroundColor = "var(--danger)";
}