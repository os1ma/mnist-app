import { initializeHandWrittingCanvas } from '../hand-writing-canvas.js'

// canvas settings

const canvas = initializeHandWrittingCanvas('#draw-area')

document.querySelector('#clear-button').addEventListener('click', () => {
  canvas.clear()
})

// submit button settings

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

  // FIXME
  const input = new Array(784)
  input.fill(0)

  const result = await predict(input)

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
