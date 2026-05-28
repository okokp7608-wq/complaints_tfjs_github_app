let rawData = [];
let metadata = {};
let model = null;
let spec = null;
let trainHistory = { loss: [], val_loss: [], acc: [], val_acc: [] };

const numericFeatures = [
  'staff_career', 'process_days', 'internal_consult', 'escalated',
  'text_length', 'attachment_count', 'satisfaction', 'resubmit', 'budget_impact'
];
const categoricalFeatures = ['channel', 'complaint_type', 'region_code'];
const target = 'priority';

const $ = (id) => document.getElementById(id);

function log(msg) {
  const box = $('logBox');
  const now = new Date().toLocaleTimeString('ko-KR');
  box.textContent += `\n[${now}] ${msg}`;
  box.scrollTop = box.scrollHeight;
}

function setStatus(kind, title, desc) {
  const badge = $('servingBadge');
  badge.className = `badge ${kind}`;
  badge.textContent = kind === 'ready' ? '서빙중' : kind === 'training' ? '학습중' : kind === 'error' ? '오류' : '대기';
  $('servingTitle').textContent = title;
  $('servingDesc').textContent = desc;
}

function fmt(n) {
  if (typeof n !== 'number') return n;
  return n.toLocaleString('ko-KR');
}

async function init() {
  await tf.ready();
  $('backendName').textContent = tf.getBackend();
  $('loadBtn').addEventListener('click', loadData);
  $('trainBtn').addEventListener('click', trainModel);
  $('saveBtn').addEventListener('click', saveModel);
  $('loadModelBtn').addEventListener('click', loadSavedModel);
  $('randomBtn').addEventListener('click', fillRandomSample);
  $('predictBtn').addEventListener('click', predictCurrent);
  drawLossChart();
}

async function loadData() {
  $('logBox').textContent = '샘플 데이터 로드 시작...';
  try {
    const [dataResp, metaResp] = await Promise.all([
      fetch('sampled_complaints.json'),
      fetch('metadata.json')
    ]);
    rawData = await dataResp.json();
    metadata = await metaResp.json();
    $('sampleRows').textContent = fmt(metadata.sample_rows);
    $('sourceRows').textContent = fmt(metadata.source_rows);
    $('classCount').textContent = Object.keys(metadata.class_counts).length;
    renderSampleMeta();
    buildSpec();
    fillSelects();
    $('trainBtn').disabled = false;
    $('randomBtn').disabled = false;
    log(`샘플 데이터 ${rawData.length.toLocaleString('ko-KR')}건 로드 완료`);
    log(`샘플링 전략: ${metadata.sample_strategy}`);
    setStatus('idle', '샘플 로드 완료', '모델 학습을 시작할 수 있습니다.');
  } catch (e) {
    setStatus('error', '데이터 로드 실패', e.message);
    log(`오류: ${e.message}`);
  }
}

function renderSampleMeta() {
  const rows = [
    ['원천 데이터 건수', fmt(metadata.source_rows)],
    ['샘플 데이터 건수', fmt(metadata.sample_rows)],
    ['샘플링 방식', metadata.sample_strategy],
    ['목표 변수', metadata.target],
    ['클래스별 샘플 수', Object.entries(metadata.class_counts).map(([k,v]) => `${k}: ${fmt(v)}`).join(' / ')],
    ['입력 변수', metadata.features.join(', ')]
  ];
  const tbody = $('sampleTable').querySelector('tbody');
  tbody.innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
}

function uniqueSorted(arr) {
  return Array.from(new Set(arr.map(v => String(v)))).sort((a, b) => a.localeCompare(b, 'ko'));
}

function buildSpec() {
  const labels = uniqueSorted(rawData.map(r => r[target]));
  const categories = {};
  categoricalFeatures.forEach(f => categories[f] = uniqueSorted(rawData.map(r => r[f])));
  const means = {}, stds = {};
  numericFeatures.forEach(f => {
    const vals = rawData.map(r => Number(r[f] ?? 0));
    const mean = vals.reduce((a,b) => a+b, 0) / vals.length;
    const variance = vals.reduce((a,b) => a + Math.pow(b - mean, 2), 0) / vals.length;
    means[f] = mean;
    stds[f] = Math.sqrt(variance) || 1;
  });
  const inputSize = numericFeatures.length + Object.values(categories).reduce((a, arr) => a + arr.length, 0);
  spec = { labels, categories, means, stds, inputSize };
  log(`입력 벡터 크기: ${inputSize}, 라벨: ${labels.join(', ')}`);
}

function fillSelects() {
  categoricalFeatures.forEach(f => {
    const el = $(f);
    el.innerHTML = spec.categories[f].map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
  });
}

function escapeHtml(s) {
  return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
}

function vectorize(row) {
  const x = [];
  numericFeatures.forEach(f => {
    const v = Number(row[f] ?? 0);
    x.push((v - spec.means[f]) / spec.stds[f]);
  });
  categoricalFeatures.forEach(f => {
    const val = String(row[f]);
    spec.categories[f].forEach(cat => x.push(val === cat ? 1 : 0));
  });
  return x;
}

function labelIndex(label) {
  return spec.labels.indexOf(String(label));
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function makeTensors() {
  const data = shuffle([...rawData]);
  const xs = data.map(vectorize);
  const ysIdx = data.map(r => labelIndex(r[target]));
  const ys = ysIdx.map(idx => spec.labels.map((_, i) => i === idx ? 1 : 0));
  const split = Math.floor(xs.length * 0.8);
  return {
    xTrain: tf.tensor2d(xs.slice(0, split)),
    yTrain: tf.tensor2d(ys.slice(0, split)),
    xVal: tf.tensor2d(xs.slice(split)),
    yVal: tf.tensor2d(ys.slice(split))
  };
}

function createModel() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape: [spec.inputSize], units: 48, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.15 }));
  m.add(tf.layers.dense({ units: 24, activation: 'relu' }));
  m.add(tf.layers.dense({ units: spec.labels.length, activation: 'softmax' }));
  m.compile({ optimizer: tf.train.adam(0.01), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

async function trainModel() {
  if (!rawData.length) return;
  setStatus('training', '모델 학습 중', '브라우저 TensorFlow.js에서 샘플링 데이터를 학습 중입니다.');
  $('trainBtn').disabled = true;
  $('predictBtn').disabled = true;
  trainHistory = { loss: [], val_loss: [], acc: [], val_acc: [] };
  model = createModel();
  const tensors = makeTensors();
  log('TensorFlow.js 모델 학습 시작...');
  try {
    await model.fit(tensors.xTrain, tensors.yTrain, {
      epochs: 25,
      batchSize: 64,
      validationData: [tensors.xVal, tensors.yVal],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          trainHistory.loss.push(logs.loss);
          trainHistory.val_loss.push(logs.val_loss);
          trainHistory.acc.push(logs.acc ?? logs.accuracy);
          trainHistory.val_acc.push(logs.val_acc ?? logs.val_accuracy);
          $('epochNow').textContent = `${epoch + 1} / 25`;
          $('trainAcc').textContent = percent(trainHistory.acc.at(-1));
          $('valAcc').textContent = percent(trainHistory.val_acc.at(-1));
          $('valLoss').textContent = trainHistory.val_loss.at(-1).toFixed(4);
          drawLossChart();
          await tf.nextFrame();
        }
      }
    });
    log('모델 학습 완료');
    setStatus('ready', 'TensorFlow.js 모델 서빙중', '브라우저 메모리 모델로 실시간 예측이 가능합니다.');
    $('saveBtn').disabled = false;
    $('predictBtn').disabled = false;
  } catch (e) {
    setStatus('error', '학습 실패', e.message);
    log(`오류: ${e.message}`);
  } finally {
    Object.values(tensors).forEach(t => t.dispose());
    $('trainBtn').disabled = false;
  }
}

function percent(v) {
  if (v == null || Number.isNaN(v)) return '-';
  return `${(v * 100).toFixed(1)}%`;
}

function drawLossChart() {
  const canvas = $('lossCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#d9e2ef';
  ctx.lineWidth = 1;
  for (let i=0; i<5; i++) {
    const y = 30 + i * 48;
    ctx.beginPath(); ctx.moveTo(45, y); ctx.lineTo(canvas.width - 20, y); ctx.stroke();
  }
  ctx.fillStyle = '#667085';
  ctx.font = '14px Malgun Gothic, sans-serif';
  ctx.fillText('Loss 추이', 45, 22);
  if (!trainHistory.loss.length) {
    ctx.fillText('학습을 시작하면 train/validation loss가 표시됩니다.', 45, 130);
    return;
  }
  const all = [...trainHistory.loss, ...trainHistory.val_loss];
  const max = Math.max(...all) || 1;
  const min = Math.min(...all) || 0;
  plotLine(ctx, trainHistory.loss, min, max, '#2556d9');
  plotLine(ctx, trainHistory.val_loss, min, max, '#d92d20');
  ctx.fillStyle = '#2556d9'; ctx.fillText('Train loss', canvas.width - 210, 22);
  ctx.fillStyle = '#d92d20'; ctx.fillText('Validation loss', canvas.width - 120, 22);
}

function plotLine(ctx, arr, min, max, color) {
  const left = 45, top = 30, w = ctx.canvas.width - 65, h = ctx.canvas.height - 60;
  ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath();
  arr.forEach((v, i) => {
    const x = left + (arr.length === 1 ? 0 : (i / (arr.length - 1)) * w);
    const y = top + (1 - (v - min) / ((max - min) || 1)) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function saveModel() {
  if (!model) return;
  const payload = { spec, metadata, savedAt: new Date().toISOString() };
  localStorage.setItem('complaints_tfjs_spec', JSON.stringify(payload));
  await model.save('indexeddb://complaints-priority-tfjs-model');
  log('IndexedDB에 모델과 전처리 spec 저장 완료');
}

async function loadSavedModel() {
  try {
    const saved = localStorage.getItem('complaints_tfjs_spec');
    if (!saved) throw new Error('저장된 전처리 spec이 없습니다. 먼저 학습 후 저장하세요.');
    spec = JSON.parse(saved).spec;
    model = await tf.loadLayersModel('indexeddb://complaints-priority-tfjs-model');
    fillSelects();
    setStatus('ready', '저장 모델 서빙중', 'IndexedDB에서 모델을 불러와 예측이 가능합니다.');
    $('predictBtn').disabled = false;
    $('saveBtn').disabled = false;
    log('IndexedDB 저장 모델 로드 완료');
  } catch (e) {
    setStatus('error', '저장 모델 로드 실패', e.message);
    log(`오류: ${e.message}`);
  }
}

function getFormRow() {
  const row = {};
  categoricalFeatures.forEach(f => row[f] = $(f).value);
  numericFeatures.forEach(f => row[f] = Number($(f).value));
  return row;
}

function fillRandomSample() {
  if (!rawData.length) return;
  const row = rawData[Math.floor(Math.random() * rawData.length)];
  categoricalFeatures.forEach(f => $(f).value = String(row[f]));
  numericFeatures.forEach(f => $(f).value = Number(row[f]));
  log(`랜덤 샘플 입력 완료. 실제 priority=${row[target]}`);
}

async function predictCurrent() {
  if (!model || !spec) return;
  const row = getFormRow();
  const x = tf.tensor2d([vectorize(row)]);
  const pred = model.predict(x);
  const probs = Array.from(await pred.data());
  x.dispose(); pred.dispose();
  const pairs = spec.labels.map((label, i) => ({ label, prob: probs[i] })).sort((a,b) => b.prob - a.prob);
  const top = pairs[0];
  $('predictionBox').innerHTML = `
    <div>예측 우선순위: <strong>${top.label}</strong> <span>(${(top.prob * 100).toFixed(1)}%)</span></div>
    ${pairs.map(p => `<div>${p.label}: ${(p.prob * 100).toFixed(1)}%<div class="bar"><span style="width:${p.prob * 100}%"></span></div></div>`).join('')}
  `;
  log(`예측 완료: ${top.label} (${(top.prob*100).toFixed(1)}%)`);
}

init();
