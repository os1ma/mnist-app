import { initializeHandWrittingCanvas } from './hand-writting-canvas.js'
import {
  getCurrentModelTag,
  getHistory,
  getModels,
  predict,
  repredict
} from './requesters.js'

const IMAGE_SIZE = 70

// canvas settings

const canvas = initializeHandWrittingCanvas('#draw-area')

document.querySelector('#clear-button').addEventListener('click', () => {
  canvas.clear()
})

// submit button settings

document.querySelector('#submit-button').addEventListener('click', async () => {
  const result = await predict(canvas)

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
  loadHistory()
})

document
  .querySelector('#submit-repredict')
  .addEventListener('click', async () => {
    await repredict()
    await loadHistory()
  })

// load initial data

getCurrentModelTag().then((tag) => {
  document.querySelector('#model-tag').textContent = tag
})

var models = []

async function loadHistoryHeader() {
  models = await getModels()

  const header = document.querySelector('#history-table-header-tr')

  while (header.firstChild) {
    header.removeChild(header.firstChild)
  }

  const td1 = document.createElement('td')
  td1.textContent = '手書き画像'
  header.appendChild(td1)

  const td2 = document.createElement('td')
  td2.textContent = 'リサイズ後'
  header.appendChild(td2)

  models.forEach((m) => {
    const td = document.createElement('td')
    td.textContent = m.tag
    header.appendChild(td)
  })
}

async function loadHistoryBody(models) {
  const { history, values } = await getHistory(models)

  const tableBody = document.querySelector('#history-table-body')

  while (tableBody.firstChild) {
    tableBody.removeChild(tableBody.firstChild)
  }

  Object.keys(values)
    .reverse()
    .forEach((imageId, i) => {
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
        const tdModelTag = document.createElement('td')
        if (values[imageId][modelTag] != undefined) {
          tdModelTag.textContent = values[imageId][modelTag]
        }
        tr.appendChild(tdModelTag)
      })
    })
}

async function loadHistory() {
  await loadHistoryHeader()
  await loadHistoryBody(models)
}

loadHistory()
