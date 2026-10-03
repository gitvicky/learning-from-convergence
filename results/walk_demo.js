(() => {
  'use strict';
  const root = document.getElementById('walk-explorer');
  if (!root) return;
  const find = id => root.querySelector(`#${id}`);
  const payload = JSON.parse(find('walk-model-data').textContent);
  const length = payload.length, maximum = 256, svgNS = 'http://www.w3.org/2000/svg';
  const startInput = find('walk-start'), countInput = find('walk-count'), modelInput = find('walk-model'), metricInput = find('walk-metric');
  const animateButton = find('walk-animate'), canvas = find('walk-paths'), map = find('walk-map'), trace = find('walk-trace');
  const state = { x: 16, count: 64, seed: 17, model: 0, metric: 0, completed: 64, fraction: 0, frame: null, start: null };
  let paths = [], running = [], modelCurve = [];

  function randomGenerator(seed) {
    let value = seed >>> 0;
    return () => {
      value += 0x6D2B79F5;
      let t = value;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function samplePaths() {
    const random = randomGenerator(state.seed);
    // Cache a complete stream; count changes select its prefix. No trajectory cutoff.
    paths = Array.from({ length: maximum }, () => {
      const positions = [state.x];
      let position = state.x;
      while (position > 0 && position < length) {
        position += random() < .5 ? -1 : 1;
        positions.push(position);
      }
      return { positions, steps: positions.length-1, right: position === length ? 1 : 0 };
    });
  }
  function modelAt(x) {
    let inputs = [x/payload.input_scale];
    for (const layer of payload.models[state.model].layers) {
      inputs = layer.weights.map((row, i) => {
        const value = row.reduce((sum, weight, j) => sum + weight*inputs[j], layer.biases[i]);
        return layer.activation === 'tanh' ? Math.tanh(value) : value;
      });
    }
    return inputs.map((value, i) => value*payload.output_scales[i]);
  }
  function exactAt(x) { return [x/length, x*(length-x)]; }
  function refreshModel() { modelCurve = Array.from({ length: length-1 }, (_,i) => modelAt(i+1)); }
  function summarize() {
    let right = 0, steps = 0;
    running = [];
    for (let i = 0; i < state.completed; i++) {
      right += paths[i].right; steps += paths[i].steps;
      running.push([right/(i+1), steps/(i+1)]);
    }
    const model = modelCurve[state.x-1], exact = exactAt(state.x);
    const format = value => value.toFixed(state.metric === 0 ? 4 : 1);
    find('walk-start-value').textContent = String(state.x);
    find('walk-count-value').textContent = String(state.count);
    find('walk-mc-value').textContent = running.length ? format(running.at(-1)[state.metric]) : '—';
    find('walk-model-value').textContent = format(model[state.metric]);
    find('walk-exact-value').textContent = format(exact[state.metric]);
    find('walk-work-summary').textContent = `${state.completed} of ${state.count} trajectories complete; ${right} exit right. These completed trajectories contain ${steps.toLocaleString('en-GB')} transitions.`;
    const visible = Math.min(8, state.completed + (state.fraction > 0 ? 1 : 0));
    find('walk-path-caption').textContent = `Most recent ${visible} paths: blue exits right, brown exits left; black is in progress.`;
    canvas.setAttribute('aria-label', `${visible} displayed walks starting at site ${state.x}; ${state.completed} complete trajectories contain ${steps} transitions, with ${right} exits at the right boundary.`);
    trace.querySelector('desc').textContent = `${state.completed} complete walks at site ${state.x}. ${state.metric === 0 ? 'Exit-right probability' : 'Mean exit time'}: MC ${running.length ? format(running.at(-1)[state.metric]) : 'pending'}, model ${format(model[state.metric])}, exact ${format(exact[state.metric])}.`;
    root.dataset.completed = String(state.completed);
    root.dataset.samples = String(state.count);
    root.dataset.site = String(state.x);
    root.dataset.right = String(right);
    root.dataset.transitions = String(steps);
    root.dataset.modelProbability = String(model[0]);
    root.dataset.modelDuration = String(model[1]);
  }
  function drawPaths() {
    const width = canvas.getBoundingClientRect().width, height = 240;
    if (width < 1) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
    const context = canvas.getContext('2d'); context.scale(ratio, ratio);
    const margin = { left: 36, right: 12, top: 12, bottom: 42 };
    const horizon = Math.max(...paths.slice(0, state.count).map(path => path.steps));
    const x = step => margin.left + step/horizon*(width-margin.left-margin.right);
    const y = site => margin.top + (1-site/length)*(height-margin.top-margin.bottom);
    context.font = '11px system-ui'; context.fillStyle = '#222'; context.lineWidth = 1;
    for (const site of [0, length/2, length]) {
      context.beginPath(); context.strokeStyle = '#ddd'; context.moveTo(x(0), y(site)); context.lineTo(x(horizon), y(site)); context.stroke();
      context.textAlign = 'right'; context.fillText(String(site), margin.left-7, y(site)+4);
    }
    context.textAlign = 'center';
    for (const step of [0, Math.round(horizon/2), horizon]) context.fillText(String(step), x(step), height-25);
    context.fillText('Steps', margin.left+(width-margin.left-margin.right)/2, height-6);
    context.save(); context.translate(10, (height-margin.bottom+margin.top)/2); context.rotate(-Math.PI/2); context.fillText('Position', 0, 0); context.restore();
    const visibleTotal = state.completed + (state.fraction > 0 && state.completed < state.count ? 1 : 0);
    for (let i = Math.max(0, visibleTotal-8); i < visibleTotal; i++) {
      const path = paths[i], partial = i === state.completed;
      const end = partial ? Math.ceil(path.steps*state.fraction) : path.steps;
      context.beginPath(); context.lineWidth = partial ? 1.8 : 1.2;
      context.strokeStyle = partial ? '#222' : path.right ? '#356aa080' : '#95532e80';
      for (let step = 0; step <= end; step++) {
        if (step) context.lineTo(x(step), y(path.positions[step])); else context.moveTo(x(0), y(path.positions[0]));
      }
      context.stroke();
      context.beginPath(); context.arc(x(end), y(path.positions[end]), 2.8, 0, 2*Math.PI);
      context.fillStyle = partial ? '#222' : path.right ? '#356aa0' : '#95532e'; context.fill();
    }
  }
  function element(tag, attributes, parent, content) {
    const item = document.createElementNS(svgNS, tag);
    for (const [key,value] of Object.entries(attributes)) item.setAttribute(key, value);
    if (content !== undefined) item.textContent = content;
    parent.append(item); return item;
  }
  function axes(svg, height, yValues, xTicks, xFraction, xLabel) {
    const width = svg.getBoundingClientRect().width;
    if (width < 1) return null;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.querySelector('g')?.remove();
    const group = element('g', {}, svg);
    const margin = { left: 44, right: 12, top: 12, bottom: 42 };
    const low = Math.min(0, ...yValues), top = Math.max(...yValues, state.metric === 0 ? 1 : 1);
    const high = top + Math.max(.05, top-low)*.07;
    const x = value => margin.left + xFraction(value)*(width-margin.left-margin.right);
    const y = value => margin.top + (high-value)/(high-low)*(height-margin.top-margin.bottom);
    element('rect', { x: margin.left, y: margin.top, width: width-margin.left-margin.right, height: height-margin.top-margin.bottom, fill: 'none', stroke: '#ccc' }, group);
    for (let i = 0; i < 4; i++) {
      const value = low + (high-low)*i/3;
      element('line', { x1: margin.left, x2: width-margin.right, y1: y(value), y2: y(value), stroke: '#eee' }, group);
      element('text', { x: margin.left-6, y: y(value)+4, 'text-anchor': 'end', 'font-size': 11, fill: '#222' }, group, value.toFixed(state.metric === 0 ? 1 : 0));
    }
    for (const value of xTicks) element('text', { x: x(value), y: height-25, 'text-anchor': value === xTicks[0] ? 'start' : value === xTicks.at(-1) ? 'end' : 'middle', 'font-size': 11, fill: '#222' }, group, String(value));
    element('text', { x: (margin.left+width-margin.right)/2, y: height-6, 'text-anchor': 'middle', 'font-size': 12, fill: '#222' }, group, xLabel);
    return { group, x, y };
  }
  function line(values, coordinates, color, dash) {
    const { group, x, y } = coordinates;
    element('path', { d: values.map(([a,b],i) => `${i ? 'L' : 'M'}${x(a).toFixed(2)},${y(b).toFixed(2)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 1.8, ...(dash ? { 'stroke-dasharray': dash } : {}) }, group);
  }
  function drawMap() {
    const exact = Array.from({ length: length-1 }, (_,i) => [i+1, exactAt(i+1)[state.metric]]);
    const learned = modelCurve.map((values,i) => [i+1, values[state.metric]]);
    const mc = running.at(-1)?.[state.metric];
    const coordinates = axes(map, 240, [...exact.map(p=>p[1]), ...learned.map(p=>p[1]), ...(mc === undefined ? [] : [mc])], [1, 16, 31], n=>(n-1)/(length-2), 'Starting site x');
    if (!coordinates) return;
    line(exact, coordinates, '#222', '2 4'); line(learned, coordinates, '#95532e', '6 4');
    if (mc !== undefined) element('circle', { cx: coordinates.x(state.x), cy: coordinates.y(mc), r: 4, fill: '#356aa0' }, coordinates.group);
    map.querySelector('desc').textContent = `Exact and learned ${state.metric === 0 ? 'exit-right probability' : 'mean exit time'} across starting sites 1 to 31. The MC point is at site ${state.x}.`;
  }
  function drawTrace() {
    const prediction = modelCurve[state.x-1][state.metric], exact = exactAt(state.x)[state.metric];
    const coordinates = axes(trace, 190, [...running.map(value=>value[state.metric]), prediction, exact], [1, 4, 16, 64, 256], n=>Math.log(n)/Math.log(maximum), 'Complete trajectories n (log scale)');
    if (!coordinates) return;
    line([[1, prediction], [maximum, prediction]], coordinates, '#95532e', '6 4');
    line([[1, exact], [maximum, exact]], coordinates, '#222', '2 4');
    if (running.length) {
      line(running.map((values,i)=>[i+1, values[state.metric]]), coordinates, '#356aa0');
      element('circle', { cx: coordinates.x(state.completed), cy: coordinates.y(running.at(-1)[state.metric]), r: 3, fill: '#356aa0' }, coordinates.group);
    }
  }
  function render() { summarize(); drawPaths(); drawMap(); drawTrace(); }
  function stop() {
    if (state.frame !== null) cancelAnimationFrame(state.frame);
    state.frame = null; state.start = null; state.fraction = 0;
    animateButton.textContent = 'Animate sampling'; animateButton.setAttribute('aria-pressed', 'false');
  }
  function complete() { stop(); state.completed = state.count; render(); }
  function animate(timestamp) {
    if (state.start === null) state.start = timestamp;
    const progress = Math.min(1, (timestamp-state.start)/5000);
    const cursor = state.count*progress;
    state.completed = Math.floor(cursor); state.fraction = cursor-state.completed; render();
    if (progress < 1) state.frame = requestAnimationFrame(animate); else { stop(); render(); }
  }
  startInput.addEventListener('input', () => { stop(); state.x = Number(startInput.value); samplePaths(); complete(); });
  countInput.addEventListener('input', () => { state.count = Number(countInput.value); complete(); });
  modelInput.addEventListener('change', () => { state.model = Number(modelInput.value); refreshModel(); render(); });
  metricInput.addEventListener('change', () => { state.metric = Number(metricInput.value); render(); });
  find('walk-resample').addEventListener('click', () => { stop(); state.seed++; samplePaths(); complete(); });
  animateButton.addEventListener('click', () => {
    if (state.frame !== null) { stop(); render(); return; }
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { complete(); return; }
    state.completed = 0; state.fraction = 0; render();
    animateButton.textContent = 'Pause sampling'; animateButton.setAttribute('aria-pressed', 'true');
    state.frame = requestAnimationFrame(animate);
  });
  samplePaths(); refreshModel(); render();
  new ResizeObserver(() => { drawPaths(); drawMap(); drawTrace(); }).observe(root);
})();
