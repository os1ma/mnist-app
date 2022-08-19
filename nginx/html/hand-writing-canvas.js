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
    this.dragging = false
    this.isEmpty = true
  }

  // public

  toBlob(mimeType, qualityArgument) {
    return new Promise((resolve) => {
      this.canvas.toBlob(
        (result) => {
          resolve(result)
        },
        mimeType,
        qualityArgument
      )
    })
  }

  clear() {
    this.isEmpty = true
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height)
  }

  // private

  dragStart() {
    this.context.beginPath()

    this.dragging = true
    this.isEmpty = false
  }

  dragEnd() {
    this.context.closePath()
    this.dragging = false

    this.lastPosition.x = null
    this.lastPosition.y = null
  }

  draw(x, y) {
    if (!this.dragging) {
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
}

export function initializeHandWrittingCanvas(selectors) {
  const canvasElement = document.querySelector(selectors)
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
    canvas.draw(event.offsetX, event.offsetY)
  })

  return canvas
}
