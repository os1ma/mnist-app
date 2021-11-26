import { initializeHandWrittingCanvas } from './hand-writting-canvas.js'
import {
  getCurrentModelTag,
  loadHistory,
  predict,
  repredict
} from './requesters.js'

// canvas settings

const canvas = initializeHandWrittingCanvas('#draw-area')

document.querySelector('#clear-button').addEventListener('click', () => {
  canvas.clear()
})

// submit button settings

document.querySelector('#submit-button').addEventListener('click', () => {
  predict(canvas)
})
document.querySelector('#submit-repredict').addEventListener('click', repredict)

// load initial data
getCurrentModelTag().then((tag) => {
  document.querySelector('#model-tag').textContent = tag
})

loadHistory()
