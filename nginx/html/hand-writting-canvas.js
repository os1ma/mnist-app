/**
 * 手書き用キャンバス
 *
 * 参考 https://tsuyopon.xyz/2018/09/14/how-to-create-drawing-app-part1/
 */
class HandWritingCanvas {
  constructor(canvas) {
    this.canvas = canvas
    this.context = canvas.getContext('2d')

    this.lastPosition = { x: null, y: null }
    this.isDrag = false
  }

  // public

  async toBlob(mimeType, qualityArgument) {
    return new Promise((resolve, reject) => {
      this.canvas.toBlob(
        (result) => {
          resolve(result)
        },
        mimeType,
        qualityArgument
      )
    })
  }

  // private

  draw(x, y) {
    if (!this.isDrag) {
      return
    }

    this.context.lineCap = 'round'
    this.context.lineJoin = 'round'
    this.context.lineWidth = 10
    this.context.strokeStyle = 'black'

    if (this.lastPosition.x === null || this.lastPosition.y === null) {
      this.context.moveTo(x, y)
    } else {
      this.context.moveTo(this.lastPosition.x, this.lastPosition.y)
    }

    this.context.lineTo(x, y)
    this.context.stroke()

    this.lastPosition.x = x
    this.lastPosition.y = y
  }

  clear() {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height)
  }

  dragStart(event) {
    this.context.beginPath()

    this.isDrag = true
  }

  dragEnd(event) {
    this.context.closePath()
    this.isDrag = false

    this.lastPosition.x = null
    this.lastPosition.y = null
  }
}

export function initializeHandWrittingCanvas(selectors) {
  const canvasElement = document.querySelector('#draw-area')
  const canvas = new HandWritingCanvas(canvasElement)

  canvasElement.addEventListener('mousedown', () => {
    canvas.dragStart()
  })
  canvasElement.addEventListener('mouseup', () => {
    canvas.dragEnd()
  })
  canvasElement.addEventListener('mouseout', () => {
    canvas.dragEnd()
  })
  canvasElement.addEventListener('mousemove', (event) => {
    canvas.draw(event.layerX, event.layerY)
  })

  return canvas
}
