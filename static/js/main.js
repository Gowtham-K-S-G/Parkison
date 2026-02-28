/**
 * Parkinson's Disease Detection System - Main JavaScript
 */

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize upload functionality
    initUploadArea();
    
    // Initialize form submission
    initUploadForm();
});

/**
 * Initialize upload area functionality
 */
function initUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');
    const imagePreview = document.getElementById('imagePreview');
    const removeImage = document.getElementById('removeImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (!uploadArea || !fileInput) return;
    
    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFileSelect(this.files[0]);
        }
    });
    
    // Handle file selection
    function handleFileSelect(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/bmp', 'image/tiff', 'image/webp'];
        
        if (!validTypes.includes(file.type)) {
            showError('Invalid file type. Please upload an image file (JPEG, PNG, BMP, TIFF, or WebP).');
            return;
        }
        
        // Validate file size (max 16MB)
        if (file.size > 16 * 1024 * 1024) {
            showError('File is too large. Maximum size is 16MB.');
            return;
        }
        
        // Preview image
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewArea.style.display = 'block';
            uploadArea.style.display = 'none';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
    
    // Remove image
    if (removeImage) {
        removeImage.addEventListener('click', function() {
            fileInput.value = '';
            imagePreview.src = '';
            previewArea.style.display = 'none';
            uploadArea.style.display = 'block';
            analyzeBtn.disabled = true;
            hideError();
        });
    }
}

/**
 * Initialize upload form submission
 */
function initUploadForm() {
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadForm) return;
    
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Check if file is selected
        if (!fileInput.files || fileInput.files.length === 0) {
            showError('Please select an image file.');
            return;
        }
        
        // Create form data
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Show loading
        loadingSpinner.style.display = 'block';
        analyzeBtn.disabled = true;
        hideError();
        
        try {
            // Send request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Store result in sessionStorage for result page
                sessionStorage.setItem('predictionResult', JSON.stringify(data));
                
                // Display result
                displayResult(data);
            } else {
                // Show error
                showError(data.error || 'An error occurred while processing the image.');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Network error. Please check your connection and try again.');
        } finally {
            // Hide loading
            loadingSpinner.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });
}

/**
 * Display result on the page
 */
function displayResult(data) {
    const resultSection = document.getElementById('resultSection');
    const uploadSection = document.getElementById('upload-section');
    
    if (!resultSection) return;
    
    // Hide upload section
    if (uploadSection) {
        uploadSection.style.display = 'none';
    }
    
    // Set result header based on prediction
    const resultHeader = document.getElementById('resultHeader');
    const predictionText = document.getElementById('predictionText');
    const confidenceText = document.getElementById('confidenceText');
    const resultImage = document.getElementById('resultImage');
    
    if (data.result_class === 'parkinson') {
        resultHeader.className = 'result-header text-center parkinson-result';
        resultHeader.innerHTML = `
            <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
            <h2>${data.result}</h2>
            <p>Analysis completed</p>
        `;
        predictionText.className = 'value text-danger';
    } else {
        resultHeader.className = 'result-header text-center healthy-result';
        resultHeader.innerHTML = `
            <i class="fas fa-check-circle fa-3x mb-3"></i>
            <h2>${data.result}</h2>
            <p>Analysis completed</p>
        `;
        predictionText.className = 'value text-success';
    }
    
    // Set values
    predictionText.textContent = data.result;
    confidenceText.textContent = data.confidence;
    resultImage.src = data.image_path;
    
    // Set probability bars
    const healthyProb = parseFloat(data.probabilities.healthy.percentage);
    const parkinsonProb = parseFloat(data.probabilities.parkinson.percentage);
    
    document.getElementById('healthyBar').style.width = healthyProb + '%';
    document.getElementById('parkinsonBar').style.width = parkinsonProb + '%';
    document.getElementById('healthyProb').textContent = data.probabilities.healthy.percentage;
    document.getElementById('parkinsonProb').textContent = data.probabilities.parkinson.percentage;
    
    // Show result section with animation
    resultSection.style.display = 'block';
    resultSection.classList.add('fade-in');
    
    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Show error message
 */
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    if (errorMessage) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }
}

/**
 * Hide error message
 */
function hideError() {
    const errorMessage = document.getElementById('errorMessage');
    if (errorMessage) {
        errorMessage.style.display = 'none';
    }
}

/**
 * Format date for display
 */
function formatDate(date) {
    const options = { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return new Date(date).toLocaleDateString('en-US', options);
}

/**
 * Animate progress bars
 */
function animateProgressBars() {
    const bars = document.querySelectorAll('.progress-bar');
    bars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0';
        setTimeout(() => {
            bar.style.width = width;
        }, 100);
    });
}

/**
 * Add smooth scrolling for anchor links
 */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Initialize animations on load
window.addEventListener('load', function() {
    // Add fade-in animation to sections
    const sections = document.querySelectorAll('.feature-card, .step-card');
    sections.forEach((section, index) => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        
        setTimeout(() => {
            section.style.opacity = '1';
            section.style.transform = 'translateY(0)';
        }, index * 100);
    });
});
