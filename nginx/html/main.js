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
}

document.querySelector('#submit-button').addEventListener('click', predict)

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
