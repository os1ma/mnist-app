export async function getCurrentModelTag() {
  const response = await axios.get('/api/models/current')
  return response.data.tag
}

export async function repredict() {
  await axios.post('/api/predictions/repredict-all')
  await loadHistory()
}

export async function predict(canvas) {
  // post image
  const headers = { 'content-type': 'multipart/form-data' }
  const data = new FormData()
  const image = await canvas.toBlob('image/png')
  data.append('image', image, 'number.png')
  const response = await axios.post('/api/predictions', data, headers)

  // show result

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
    tdMessage.textContent = i
    tr.appendChild(tdMessage)

    // 確率
    const tdCreatedAt = document.createElement('td')
    tdCreatedAt.textContent = (v * 100).toFixed(1)
    tr.appendChild(tdCreatedAt)

    tableBody.appendChild(tr)
  })

  canvas.clear()
  loadHistory()
}

var models = []

async function loadModels() {
  const response = await axios.get('/api/models')
  models = response.data.models
  console.log('=== models ===')
  console.log(models)

  const header = document.querySelector('#history-table-header-tr')

  // TODO 意味を明確にする
  while (header.childNodes.length > 4) {
    console.log('removing header last...')
    console.log(header.lastChild)
    header.removeChild(header.lastChild)
  }

  models.forEach((m) => {
    const td = document.createElement('td')
    td.textContent = m.tag
    header.appendChild(td)
  })
}

const IMAGE_SIZE = 70

function calculateHighProbabilityValue(result) {
  var maxResult = 0
  var maxResultIndex = 0
  result.forEach((v, i) => {
    if (v > maxResult) {
      maxResult = v
      maxResultIndex = i
    }
  })
  return maxResultIndex
}

export async function loadHistory() {
  await loadModels()

  const response = await axios.get('/api/prediction-history')
  const history = response.data.history
  console.log('=== history ===')
  console.log(history)

  const tableBody = document.querySelector('#history-table-body')

  while (tableBody.firstChild) {
    tableBody.removeChild(tableBody.firstChild)
  }

  const values = {}
  history.forEach((h) => {
    // 値を計算
    models.forEach((m) => {
      const imageId = h.image_id
      const modelTag = h.model_tag

      if (!values[imageId]) {
        values[imageId] = {}
      }

      const result = JSON.parse(h.result)
      if (!values[imageId][modelTag]) {
        values[imageId][modelTag] = calculateHighProbabilityValue(result)
      }
    })
  })
  console.log('=== values ===')
  console.log(values)

  Object.keys(values)
    .reverse()
    .forEach((imageId, i) => {
      console.log(`=== valeus[imageId] imageId = ${imageId} ===`)
      console.log(values[imageId])

      const tr = document.createElement('tr')
      tableBody.appendChild(tr)

      const h = history.filter((h) => h.image_id == imageId)[0]

      // 手書き画像
      const originalImage = document.createElement('img')
      originalImage.src = h.original_image_path
      originalImage.height = IMAGE_SIZE
      const tdOriginalImage = document.createElement('td')
      tdOriginalImage.appendChild(originalImage)
      tr.appendChild(tdOriginalImage)

      // リサイズ後
      const resizedImage = document.createElement('img')
      resizedImage.src = h.preprocessed_image_path
      resizedImage.height = IMAGE_SIZE
      const tdResizedImage = document.createElement('td')
      tdResizedImage.appendChild(resizedImage)
      tr.appendChild(tdResizedImage)

      // 値
      models.forEach((m) => {
        const modelTag = m.tag
        console.log(`modelTag = ${modelTag}`)
        const tdModelTag = document.createElement('td')
        if (values[imageId][modelTag] != undefined) {
          tdModelTag.textContent = values[imageId][modelTag]
        }
        tr.appendChild(tdModelTag)
      })
    })
}
