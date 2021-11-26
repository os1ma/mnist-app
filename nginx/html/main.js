async function promiseCanvasToBlob(mimeType, qualityArgument) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (result) => {
        resolve(result)
      },
      mimeType,
      qualityArgument
    )
  })
}

async function predict() {
  // post image
  const headers = { 'content-type': 'multipart/form-data' }
  const data = new FormData()
  const image = await promiseCanvasToBlob('image/png')
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

  clear()
  loadHistory()
}

async function getModelTag() {
  const response = await axios.get('/api/models/current')
  const tag = response.data.tag

  document.querySelector('#model-tag').textContent = tag
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

async function loadHistory() {
  await loadModels()

  const response = await axios.get('/api/prediction-history')
  const history = response.data.history
  console.log('=== history ===')
  console.log(history)

  const tableBody = document.querySelector('#history-table-body')

  while (tableBody.firstChild) {
    tableBody.removeChild(tableBody.firstChild)
  }

  history.forEach((h) => {
    const tr = document.createElement('tr')

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

    // TODO
    const tdModelTag = document.createElement('td')
    const result = JSON.parse(h.result)
    var maxResult = 0
    var maxResultIndex = 0
    result.forEach((v, i) => {
      if (v > maxResult) {
        maxResult = v
        maxResultIndex = i
      }
    })
    tdModelTag.textContent = maxResultIndex
    tr.appendChild(tdModelTag)

    tableBody.appendChild(tr)
  })
}

async function repredict() {
  await axios.post('/api/predictions/repredict-all')
  await loadHistory()
}

document.querySelector('#submit-button').addEventListener('click', predict)
document.querySelector('#submit-repredict').addEventListener('click', repredict)

getModelTag()

loadHistory()

// canvas
// see https://tsuyopon.xyz/2018/09/14/how-to-create-drawing-app-part1/

const canvas = document.querySelector('#draw-area')
const context = canvas.getContext('2d')

const lastPosition = { x: null, y: null }
let isDrag = false

function draw(x, y) {
  if (!isDrag) {
    return
  }

  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.lineWidth = 10
  context.strokeStyle = 'black'

  if (lastPosition.x === null || lastPosition.y === null) {
    context.moveTo(x, y)
  } else {
    context.moveTo(lastPosition.x, lastPosition.y)
  }

  context.lineTo(x, y)
  context.stroke()

  lastPosition.x = x
  lastPosition.y = y
}

function clear() {
  context.clearRect(0, 0, canvas.width, canvas.height)
}

function dragStart(event) {
  context.beginPath()

  isDrag = true
}

function dragEnd(event) {
  context.closePath()
  isDrag = false

  lastPosition.x = null
  lastPosition.y = null
}

const clearButton = document.querySelector('#clear-button')
clearButton.addEventListener('click', clear)

canvas.addEventListener('mousedown', dragStart)
canvas.addEventListener('mouseup', dragEnd)
canvas.addEventListener('mouseout', dragEnd)
canvas.addEventListener('mousemove', (event) => {
  draw(event.layerX, event.layerY)
})
