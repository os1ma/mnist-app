export async function getCurrentModelTag() {
  const response = await axios.get('/api/models/current')
  return response.data.tag
}

export async function repredict() {
  await axios.post('/api/predictions/repredict-all')
}

export async function predict(blob) {
  const headers = { 'content-type': 'multipart/form-data' }

  const data = new FormData()
  data.append('image', blob, 'number.png')

  const response = await axios.post('/api/predictions', data, headers)
  return response.data.result
}

export async function getModels() {
  const response = await axios.get('/api/models')
  return response.data.models
}

export async function getHistory(models) {
  const response = await axios.get('/api/prediction-history')
  const history = response.data.history

  const values = {}
  history.forEach((h) => {
    // 値を計算
    models.forEach((m) => {
      const imageId = h.imageId
      const modelTag = h.modelTag

      if (!values[imageId]) {
        values[imageId] = {}
      }

      const result = JSON.parse(h.result)
      if (!values[imageId][modelTag] && result != null) {
        values[imageId][modelTag] = calculateHighProbabilityValue(result)
      }
    })
  })

  return { history, values }
}

// private

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
