import { initializeHandWrittingCanvas } from '../hand-writing-canvas.js'

// canvas settings

const canvas = initializeHandWrittingCanvas('#draw-area')

document.querySelector('#clear-button').addEventListener('click', () => {
  canvas.clear()
})

// submit button settings

function normalize(arr) {
  // FIXME
  return arr.map((v) => v / 128 - 1)
}

async function preprocess(blob) {
  // resize

  const canvas = document.createElement('canvas')

  document.querySelector('body').appendChild(canvas)

  const ctx = canvas.getContext('2d')
  canvas.height = 28
  canvas.width = 28
  ctx.drawImage

  const bitmap = await createImageBitmap(blob, {
    resizeHeight: 28,
    resizeHeight: 28
  })
  ctx.drawImage(bitmap, 0, 0)
  const imageData = ctx.getImageData(0, 0, 28, 28)

  // get 28 * 28 elem

  const dst = []
  for (let i = 0; i < imageData.data.length; i++) {
    if (i % 4 === 3) {
      const current = imageData.data[i]
      dst.push(current)
    }
  }

  const normalized = normalize(dst)

  return normalized
}

function softmax(arr) {
  return arr.map((value) => {
    return Math.exp(value) / arr.map((y) => Math.exp(y)).reduce((a, b) => a + b)
  })
}

async function predict(input) {
  const session = await ort.InferenceSession.create('./v1.onnx')

  const inputName = session.inputNames[0]

  const feeds = {}
  feeds[inputName] = new ort.Tensor('float32', input)

  const results = await session.run(feeds)

  const data = results.output.data

  const softmaxed = softmax(data)

  return softmaxed
}

document.querySelector('#submit-button').addEventListener('click', async () => {
  if (canvas.isEmpty) {
    return
  }

  // 推論実行

  const blob = await canvas.toBlob('image/png')

  const preprocessed = await preprocess(blob)

  const result = await predict(preprocessed)

  // 推論結果の画像を表示

  const imageUrl = URL.createObjectURL(blob)
  document.querySelector('#result-image').src = imageUrl

  // 推論結果のテーブル内を表示

  const tableBody = document.querySelector('#result-table-body')

  while (tableBody.firstChild) {
    tableBody.removeChild(tableBody.firstChild)
  }

  result.forEach((v, i) => {
    const tr = document.createElement('tr')

    // 数字
    const tdNumber = document.createElement('td')
    tdNumber.textContent = i
    tr.appendChild(tdNumber)

    // 確率
    const tdProbability = document.createElement('td')
    tdProbability.textContent = (v * 100).toFixed(1)
    tr.appendChild(tdProbability)

    tableBody.appendChild(tr)
  })

  canvas.clear()
})
