// --- Element Selectors ---
const predictionForm = document.getElementById('prediction-form');
const resultContainer = document.getElementById('result-container');
const predictionText = document.getElementById('prediction-text');
const confidenceText = document.getElementById('confidence-text');
const classifyBtn = document.getElementById('classify-btn');

const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');
const chatWindow = document.getElementById('chat-window');

// --- API URLs ---
// IMPORTANT: Replace this with your actual Render API URL
const API_URL = 'https://exoplanet-api.onrender.com'; 

// --- State Management ---
let predictionContext = null; // This will store the data for the chatbot

// --- Prediction Form Logic ---
predictionForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    classifyBtn.textContent = 'Analyzing...';
    classifyBtn.disabled = true;

    // Collect data from all input fields
    const formData = {
        'koi_fpflag_co': parseFloat(document.getElementById('koi_fpflag_co').value),
        'koi_fpflag_ss': parseFloat(document.getElementById('koi_fpflag_ss').value),
        'koi_period': parseFloat(document.getElementById('koi_period').value),
        'koi_duration': parseFloat(document.getElementById('koi_duration').value),
        'koi_depth': parseFloat(document.getElementById('koi_depth').value),
        // Add ALL other form fields here to match your model
    };

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData),
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const result = await response.json();

        // Display result
        predictionText.textContent = `Prediction: ${result.prediction}`;
        confidenceText.textContent = `Confidence: ${result.confidence}`;
        resultContainer.classList.remove('hidden');

        // *** IMPORTANT: Store context for the chatbot ***
        predictionContext = {
            input_data: formData,
            prediction: result.prediction,
            confidence: result.confidence
        };

        // Enable the chat
        chatInput.disabled = false;
        chatSendBtn.disabled = false;
        addBotMessage("I've analyzed the data! What would you like to know about this candidate?");

    } catch (error) {
        console.error('Error:', error);
        predictionText.textContent = 'Error making prediction.';
        confidenceText.textContent = '';
    } finally {
        classifyBtn.textContent = 'Classify Candidate';
        classifyBtn.disabled = false;
    }
});

// --- Chat Form Logic ---
chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const userMessage = chatInput.value.trim();
    if (!userMessage) return;

    // Display user message and clear input
    addUserMessage(userMessage);
    chatInput.value = '';

    if (!predictionContext) {
        addBotMessage("Please classify a candidate first before asking questions.");
        return;
    }
    
    // Show loading indicator
    const loadingIndicator = addLoadingMessage();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userMessage,
                context: predictionContext
            }),
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const result = await response.json();

        // Remove loading indicator and add bot's reply
        loadingIndicator.remove();
        addBotMessage(result.reply);

    } catch (error) {
        console.error('Chat Error:', error);
        loadingIndicator.remove();
        addBotMessage("Sorry, I'm having trouble connecting to my knowledge base. Please try again.");
    }
});

// --- Chat Helper Functions ---
function addUserMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'user-message';
    messageElement.innerHTML = `<p>${message}</p>`;
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addBotMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'bot-message';
    messageElement.innerHTML = `<p>${message}</p>`; // Use innerHTML to render formatting from Gemini if any
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addLoadingMessage() {
    const loadingElement = document.createElement('div');
    loadingElement.className = 'bot-message loading-message';
    loadingElement.innerHTML = `<p><i>Exo-Chat is thinking...</i></p>`;
    chatWindow.appendChild(loadingElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return loadingElement;
}