function escapeHTML(obj) {
  return JSON.stringify(obj)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

async function predict() {
  const response = await axios.post('/api/predict')
  const result = response.data.result
  console.log(`result = ${result}`)

  const tableBody = document.querySelector('#result-table-body')

  while (tableBody.firstChild) {
    tableBody.removeChild(tableBody.firstChild)
  }

  result.forEach((v, i) => {
    const tr = document.createElement('tr')

    // 数字
    const tdMessage = document.createElement('td')
    tdMessage.textContent = escapeHTML(i)
    tr.appendChild(tdMessage)

    // 確率
    const tdCreatedAt = document.createElement('td')
    tdCreatedAt.textContent = escapeHTML(v)
    tr.appendChild(tdCreatedAt)

    tableBody.appendChild(tr)
  })
}

document.querySelector('#submit-button').addEventListener('click', predict)
