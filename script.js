// Theme toggling functionality
const themeToggle = document.getElementById('themeToggle');
let theme = 'default';

themeToggle.addEventListener('click', () => {
    if (theme === 'default') {
        document.body.style.background = 'linear-gradient(to bottom, teal, lightblue, lightgray)';
        theme = 'theme1';
    } else if (theme === 'theme1') {
        document.body.style.background = 'linear-gradient(to bottom, #C2CADO, #C2B9B0, #7685A1)';
        theme = 'theme2';
    } else {
        document.body.style.background = 'linear-gradient(to bottom, #0e2744, #d7eef9)';
        theme = 'default';
    }
});

// Function to speak a message
function speakMessage(message) {
    const speech = new SpeechSynthesisUtterance(message);
    speech.lang = 'en-US';
    window.speechSynthesis.speak(speech);
}

// Suggestion box toggle functionality
const suggestionIcon = document.getElementById('suggestionIcon');
const suggestionBox = document.getElementById('suggestionBox');
const submitSuggestion = document.getElementById('submitSuggestion');
const feedbackMessage = document.getElementById('feedbackMessage');

suggestionIcon.addEventListener('click', () => {
    suggestionBox.style.display = suggestionBox.style.display === 'block' ? 'none' : 'block';
});

// Handle suggestion box submission
submitSuggestion.addEventListener('click', () => {
    const email = document.getElementById('email').value.trim();
    const phone = document.getElementById('phone').value.trim();
    const feedback = document.getElementById('feedback').value.trim();

    if (!email || !phone || !feedback) {
        feedbackMessage.textContent = 'Please fill out all fields.';
        feedbackMessage.style.color = 'red';
        return;
    }

    // Show "Submitted" message
    feedbackMessage.textContent = 'Submitted! Thank you for your feedback.';
    feedbackMessage.style.color = 'green';

    // Clear fields after 2 seconds
    setTimeout(() => {
        document.getElementById('email').value = '';
        document.getElementById('phone').value = '';
        document.getElementById('feedback').value = '';
        feedbackMessage.textContent = '';
    }, 2000);
});

// Handle file upload and prediction for images
const submitBtn = document.getElementById('submitBtn');
const resultDiv = document.getElementById('result');
const uploadedImage = document.getElementById('uploadedImage');

submitBtn.addEventListener('click', async () => {
    const fileInput = document.getElementById('fileUpload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        uploadedImage.src = e.target.result;
        uploadedImage.style.display = 'block';
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:5000/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        if (data.error) {
            resultDiv.innerHTML = `Error: ${data.error}`;
        } else {
            resultDiv.innerHTML = `Predicted Action: ${data.action}`;

            // Only speak "Concentrate on driving" if action is not "Safe Driving"
            if (data.action !== 'Safe Driving') {
                speakMessage('Concentrate on driving.');
            }
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.innerHTML = 'Error occurred while predicting.';
    }
});
