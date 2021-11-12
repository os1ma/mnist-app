function escapeHTML(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

async function predict() {
  const response = await axios.post('/api/predict')
  const result = response.data.result

  const resultElement = document.querySelector('#result').value
  resultElement.textContent = escapeHTML(result)
}

document.querySelector('#submit-button').addEventListener('click', predict)
