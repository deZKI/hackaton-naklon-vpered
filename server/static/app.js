const video1 = document.getElementById('video1');
const video2 = document.getElementById('video2');
const canvas1 = document.getElementById('canvas1');
const canvas2 = document.getElementById('canvas2');
const ctx1 = canvas1.getContext('2d');
const ctx2 = canvas2.getContext('2d');
const counter = document.getElementById('counter');

canvas1.width = canvas2.width = 320;
canvas1.height = canvas2.height = 240;

const ws = new WebSocket('ws://localhost:8000/ws/inference');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  counter.innerText = data.pushups;

  // отрисовка скелета на canvas1
  ctx1.clearRect(0, 0, canvas1.width, canvas1.height);
  if (data.skeleton && data.scores) {
    data.skeleton.forEach(([x, y], i) => {
      if (data.scores[i] > 0.5) {
        ctx1.beginPath();
        ctx1.arc(x, y, 4, 0, 2 * Math.PI);
        ctx1.fillStyle = 'red';
        ctx1.fill();
      }
    });
  }
};

async function getCameras() {
  await navigator.mediaDevices.getUserMedia({ video: true });
  const devices = await navigator.mediaDevices.enumerateDevices();
  console.log(devices)
  const videoDevices = devices.filter(d => d.kind === 'videoinput');
  console.log(videoDevices.length)
  if (videoDevices.length < 2) {
    alert('Нужно минимум 2 видеокамеры');
    return;
  }

  const stream1 = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: videoDevices[0].deviceId }
  });
  const stream2 = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: videoDevices[1].deviceId }
  });

  video1.srcObject = stream1;
  video2.srcObject = stream2;

  setInterval(() => {
    ctx1.drawImage(video1, 0, 0, canvas1.width, canvas1.height);
    ctx2.drawImage(video2, 0, 0, canvas2.width, canvas2.height);

    Promise.all([
      new Promise(resolve => canvas1.toBlob(resolve, 'image/jpeg')),
      new Promise(resolve => canvas2.toBlob(resolve, 'image/jpeg')),
    ]).then(([blob1, blob2]) => {
      Promise.all([blob1.arrayBuffer(), blob2.arrayBuffer()]).then(([buf1, buf2]) => {
        // отправка как JSON+Base64 или сериализованный бинарник
        ws.send(JSON.stringify({
          cam1: Array.from(new Uint8Array(buf1)),
          cam2: Array.from(new Uint8Array(buf2))
        }));
      });
    });
  }, 1);
}

getCameras();
