import { useState, useEffect } from 'react';

type Prediction = { label: string; confidence: number };
type AttackResponse = {
  success: boolean;
  message?: string;
  original_prediction?: Prediction;
  adversarial_prediction?: Prediction;
  original_top5?: Prediction[];
  adversarial_top5?: Prediction[];
  images?: { original: string; adversarial: string; noise: string };
};
type DenoiseResponse = {
  denoised_image: string;
  original_prediction: Prediction;
  restored_prediction: Prediction;
  psnr_score: number;
  ssim_score: number;
  success: boolean;
};
type DetectResponse = {
  is_adversarial: boolean;
  confidence: number;
  votes: number;
  verdict: string;
  prediction: Prediction;
  scores: {
    gaussian_blur_delta:  number;
    bit_depth_delta:      number;
    jpeg_compress_delta:  number;
    max_delta:            number;
  };
  features:  number[];
  threshold: number;
  delta_blur: number;
  delta_bits: number;
  delta_jpeg: number;
  delta_max:  number;
};
type ImageNetClass = { id: string; name: string; raw: string };

const DENOISE_METHODS = [
  { value: 'tv',                   label: 'Total Variation (TV)' },
  { value: 'gaussian',             label: 'Gaussian Smoothing' },
  { value: 'jpeg',                 label: 'JPEG Compression' },
  { value: 'feature_squeezing',    label: 'Feature Squeezing' },
  { value: 'randomized_smoothing', label: 'Randomized Smoothing' },
];

function App() {
  const [activeTab, setActiveTab] = useState<'attack' | 'denoise' | 'detect'>('attack');

  // ── Attack state ──────────────────────────────────────────────────────────
  const [classes, setClasses]     = useState<ImageNetClass[]>([]);
  const [file, setFile]           = useState<File | null>(null);
  const [preview, setPreview]     = useState<string | null>(null);
  const [targetClass, setTargetClass] = useState<string>('gibbon');
  const [epsilon, setEpsilon]     = useState<number>(0.05);
  const [iterations, setIterations] = useState<number>(5);
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState<AttackResponse | null>(null);
  const [error, setError]         = useState<string | null>(null);

  // ── Denoise state ─────────────────────────────────────────────────────────
  const [dFile, setDFile]         = useState<File | null>(null);
  const [dPreview, setDPreview]   = useState<string | null>(null);
  const [dMethod, setDMethod]     = useState<string>('tv');
  const [dEpsilon, setDEpsilon]   = useState<number>(0.05);
  const [dLoading, setDLoading]   = useState(false);
  const [dResult, setDResult]     = useState<DenoiseResponse | null>(null);
  const [dError, setDError]       = useState<string | null>(null);

  // ── Detect state ──────────────────────────────────────────────────────────
  const [dtFile, setDtFile]       = useState<File | null>(null);
  const [dtPreview, setDtPreview] = useState<string | null>(null);
  const [dtEpsilon, setDtEpsilon] = useState<number>(0.05);
  const [dtLoading, setDtLoading] = useState(false);
  const [dtResult, setDtResult]   = useState<DetectResponse | null>(null);
  const [dtError, setDtError]     = useState<string | null>(null);


  useEffect(() => {
    fetch('http://localhost:8000/classes')
      .then(r => r.json())
      .then(d => setClasses(Array.isArray(d) ? d : []))
      .catch(() => setClasses([]));
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const f = e.target.files[0];
      setFile(f); setPreview(URL.createObjectURL(f));
      setResult(null); setError(null);
    }
  };

  /**
   * Download an image by sending it to the backend /download endpoint.
   * The server returns it with Content-Disposition: attachment; filename=...
   * which every browser respects — no data URL / blob URL quirks.
   */
  const downloadBase64Image = async (dataUrl: string, filename: string) => {
    try {
      const res = await fetch('http://localhost:8000/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data_url: dataUrl, filename }),
      });
      if (!res.ok) throw new Error(`Download failed: ${res.status}`);

      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = filename;   // redundant but keeps DevTools happy
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 200);
    } catch (err) {
      console.error('Download error:', err);
      alert('Download failed. Please try again.');
    }
  };


  const handleDetect = async () => {
    if (!dtFile) { setDtError('Please upload an image first.'); return; }
    setDtLoading(true); setDtError(null); setDtResult(null);
    const fd = new FormData();
    fd.append('image', dtFile); fd.append('epsilon', dtEpsilon.toString());
    try {
      const res = await fetch('http://localhost:8000/detect', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      setDtResult(await res.json());
    } catch (err: any) { setDtError(err.toString()); }
    finally { setDtLoading(false); }
  };

  const handleAttack = async () => {
    if (!file) { setError('Please upload an image first.'); return; }
    setLoading(true); setError(null);
    const sel = classes.find(c => c.name === targetClass || c.id === targetClass);
    const tv  = sel ? sel.raw : targetClass;
    const fd  = new FormData();
    fd.append('image', file); fd.append('target_class', tv);
    fd.append('epsilon', epsilon.toString()); fd.append('iterations', iterations.toString());
    try {
      const res = await fetch('http://localhost:8000/attack', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data: AttackResponse = await res.json();
      setResult(data);
      if (!data.success && data.message) setError(data.message);
    } catch (err: any) { setError(err.toString()); }
    finally { setLoading(false); }
  };

  const handleDenoise = async () => {
    if (!dFile) { setDError('Please upload an adversarial image first.'); return; }
    setDLoading(true); setDError(null); setDResult(null);
    const fd = new FormData();
    fd.append('image', dFile); fd.append('method', dMethod);
    fd.append('epsilon', dEpsilon.toString());
    try {
      const res = await fetch('http://localhost:8000/denoise', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      setDResult(await res.json());
    } catch (err: any) { setDError(err.toString()); }
    finally { setDLoading(false); }
  };

  const renderBarChart = (title: string, predictions: Prediction[]) => (
    <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 flex-1">
      <h3 className="font-semibold text-gray-700 mb-4">{title}</h3>
      <div className="space-y-3">
        {predictions.map((p, i) => (
          <div key={i} className="relative">
            <div className="flex justify-between text-xs text-gray-600 mb-1 px-1">
              <span className="font-medium truncate pr-2 max-w-[70%]">{p.label}</span>
              <span>{p.confidence.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-100 rounded-lg h-5 overflow-hidden">
              <div className="bg-indigo-500 h-full transition-all duration-700 rounded-lg" style={{ width: `${p.confidence}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const MetricBadge = ({ label, value, unit }: { label: string; value: number; unit?: string }) => (
    <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-4 text-center">
      <div className="text-xs text-indigo-400 font-semibold uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-bold text-indigo-700">{value.toFixed(2)}<span className="text-sm font-normal ml-1 text-indigo-400">{unit}</span></div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans pb-20">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Adversarial Attack Explorer</h1>
          <p className="mt-2 text-gray-500 max-w-2xl leading-relaxed">
            See how small, imperceptible changes to an image can completely fool a powerful AI model.
            Uses the <strong>Iterative Target Class Method</strong> on ResNet-18, with built-in defenses.
          </p>
          {/* Tab bar */}
          <div className="flex gap-1 mt-6 bg-gray-100 p-1 rounded-xl w-fit">
            {(['attack', 'denoise', 'detect'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-5 py-2 rounded-lg text-sm font-semibold transition-all capitalize ${
                  activeTab === tab
                    ? 'bg-white shadow text-indigo-700'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab === 'attack' ? '⚡ Attack' : tab === 'denoise' ? '🛡️ Denoise' : '🔍 Detect'}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-8 mt-4">

        {/* ════════════════════ ATTACK TAB ════════════════════ */}
        {activeTab === 'attack' && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
              {/* Upload & Select */}
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                <h2 className="text-lg font-semibold text-gray-800 mb-4">1. Setup Attack</h2>
                <div className="mb-5">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Upload Image (JPG/PNG)</label>
                  <label htmlFor="attack-file" className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-xl cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
                    {preview
                      ? <img src={preview} className="h-24 w-auto object-contain rounded border border-gray-200" />
                      : <span className="text-sm text-gray-400 font-medium">Click to upload image</span>}
                    <input id="attack-file" type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
                  </label>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Target Class to Fool the AI</label>
                  <select className="w-full rounded-lg border border-gray-300 p-2.5 text-sm bg-white focus:ring-2 focus:ring-indigo-500" value={targetClass} onChange={e => setTargetClass(e.target.value)}>
                    <option value="gibbon">gibbon (Default)</option>
                    <option value="goldfish">goldfish</option>
                    {classes.filter(c => c.name !== 'gibbon' && c.name !== 'goldfish').map(c => (
                      <option key={c.id} value={c.name}>{c.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Parameters */}
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex flex-col">
                <h2 className="text-lg font-semibold text-gray-800 mb-4">2. Configure Parameters</h2>
                <div className="mb-6 flex-1">
                  <div className="flex justify-between items-end mb-1">
                    <label className="text-sm font-medium text-gray-700">Epsilon <span className="text-xs text-gray-400">(Noise Strength)</span></label>
                    <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded text-gray-600">{epsilon}</span>
                  </div>
                  <input type="range" min="0.001" max="0.1" step="0.001" value={epsilon} onChange={e => setEpsilon(parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                </div>
                <div className="mb-6 flex-1">
                  <div className="flex justify-between items-end mb-1">
                    <label className="text-sm font-medium text-gray-700">Iterations</label>
                    <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded text-gray-600">{iterations}</span>
                  </div>
                  <input type="range" min="1" max="20" step="1" value={iterations} onChange={e => setIterations(parseInt(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                </div>
                <button onClick={handleAttack} disabled={loading || !file} className="mt-auto w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-4 rounded-xl shadow transition-colors disabled:bg-indigo-300 flex items-center justify-center gap-2">
                  {loading && <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/></svg>}
                  {loading ? 'Generating Attack...' : 'Run Attack'}
                </button>
              </div>
            </div>

            {error && (
              <div className="mb-8 p-4 bg-yellow-50 border border-yellow-200 rounded-xl text-yellow-800 text-sm font-medium">{error}</div>
            )}

            {result?.success && result.images && (
              <div>
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-gray-900">Attack Results</h2>
                  <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium border border-green-200">Attack Successful</div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                  <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100">
                    <div className="bg-gray-100 aspect-square flex items-center justify-center p-4">
                      <img src={result.images.original} className="max-w-full max-h-full object-contain rounded drop-shadow-md" />
                    </div>
                    <div className="p-4 border-t border-gray-100">
                      <h4 className="text-xs uppercase text-gray-400 font-semibold tracking-wider mb-1">Original Prediction</h4>
                      <div className="font-semibold text-lg text-gray-900 truncate">{result.original_prediction?.label}</div>
                      <div className="text-sm text-gray-500">{result.original_prediction?.confidence.toFixed(2)}% Confidence</div>
                    </div>
                  </div>
                  <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100">
                    <div className="bg-gray-900 aspect-square flex items-center justify-center p-4">
                      <img src={result.images.noise} className="max-w-full max-h-full object-contain mix-blend-screen opacity-90" />
                    </div>
                    <div className="p-4 border-t border-gray-100">
                      <h4 className="text-xs uppercase text-gray-400 font-semibold tracking-wider mb-1">Added Noise (Amplified)</h4>
                      <div className="text-sm text-gray-500 mt-1">Bounded by ε={epsilon}</div>
                    </div>
                  </div>
                  <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100 ring-2 ring-indigo-500">
                    <div className="bg-gray-100 aspect-square flex items-center justify-center p-4">
                      <img src={result.images.adversarial} className="max-w-full max-h-full object-contain rounded drop-shadow-md" />
                    </div>
                    <div className="p-4 border-t border-gray-100 bg-indigo-50">
                      <h4 className="text-xs uppercase text-indigo-400 font-semibold tracking-wider mb-1">Fooled Prediction</h4>
                      <div className="font-semibold text-lg text-indigo-900 truncate">{result.adversarial_prediction?.label}</div>
                      <div className="text-sm text-indigo-600 font-medium mb-3">{result.adversarial_prediction?.confidence.toFixed(2)}% Confidence</div>
                      <button
                        onClick={() => downloadBase64Image(
                          result.images.adversarial,
                          `adversarial_${targetClass.replace(/\s+/g, '_')}.png`
                        )}
                        className="flex items-center justify-center gap-2 w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-semibold transition-all"
                      >
                        ↓ Download Fooled Image
                      </button>
                    </div>
                  </div>
                </div>
                <div className="flex flex-col md:flex-row gap-6 mt-8">
                  {result.original_top5 && renderBarChart('Original Top-5', result.original_top5)}
                  {result.adversarial_top5 && renderBarChart('Adversarial Top-5', result.adversarial_top5)}
                </div>
              </div>
            )}
          </>
        )}

        {/* ════════════════════ DENOISE TAB ════════════════════ */}
        {activeTab === 'denoise' && (
          <>
            <div className="mb-6">
              <p className="text-gray-500 text-sm leading-relaxed max-w-2xl">
                Upload an adversarial image and apply a defense strategy to attempt to recover the correct classification.
                Compare PSNR &amp; SSIM to measure how much image quality was preserved.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
              {/* Upload */}
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                <h2 className="text-lg font-semibold text-gray-800 mb-4">1. Upload Adversarial Image</h2>
                <label htmlFor="denoise-file" className="flex flex-col items-center justify-center w-full h-36 border-2 border-dashed border-violet-300 rounded-xl cursor-pointer bg-violet-50 hover:bg-violet-100 transition-colors">
                  {dPreview
                    ? <img src={dPreview} className="h-28 w-auto object-contain rounded border border-violet-200" />
                    : <>
                        <span className="text-3xl mb-2">🎯</span>
                        <span className="text-sm text-violet-500 font-medium">Click to upload adversarial image</span>
                      </>}
                  <input id="denoise-file" type="file" accept="image/*" className="hidden" onChange={e => {
                    if (e.target.files?.[0]) {
                      const f = e.target.files[0];
                      setDFile(f); setDPreview(URL.createObjectURL(f));
                      setDResult(null); setDError(null);
                    }
                  }} />
                </label>
              </div>

              {/* Method & Run */}
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex flex-col gap-5">
                <h2 className="text-lg font-semibold text-gray-800">2. Configure Defense</h2>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Denoising Method</label>
                  <select id="denoise-method" value={dMethod} onChange={e => setDMethod(e.target.value)} className="w-full rounded-lg border border-gray-300 p-2.5 text-sm bg-white focus:ring-2 focus:ring-violet-500 focus:border-violet-500">
                    {DENOISE_METHODS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                  </select>
                  <div className="mt-2 text-xs text-gray-400">
                    {dMethod === 'tv' && 'Minimizes total variation to smooth structured noise while preserving edges.'}
                    {dMethod === 'gaussian' && 'Applies a Gaussian low-pass filter to blur out high-frequency perturbations.'}
                    {dMethod === 'jpeg' && 'Round-trips the image through JPEG compression to destroy adversarial artifacts.'}
                    {dMethod === 'feature_squeezing' && 'Quantizes pixel bit depth to round away small adversarial values.'}
                    {dMethod === 'randomized_smoothing' && 'Averages multiple noisy copies to suppress the structured adversarial signal.'}
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-end mb-1">
                    <label className="text-sm font-medium text-gray-700">Attack Epsilon <span className="text-xs text-gray-400">(used during attack)</span></label>
                    <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded text-gray-600">{dEpsilon}</span>
                  </div>
                  <input type="range" min="0.001" max="0.1" step="0.001" value={dEpsilon} onChange={e => setDEpsilon(parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-600" />
                </div>

                <button onClick={handleDenoise} disabled={dLoading || !dFile} className="mt-auto w-full bg-violet-600 hover:bg-violet-700 text-white font-medium py-3 px-4 rounded-xl shadow transition-colors disabled:bg-violet-300 flex items-center justify-center gap-2">
                  {dLoading && <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/></svg>}
                  {dLoading ? 'Denoising...' : '🛡️ Run Defense'}
                </button>
              </div>
            </div>

            {dError && (
              <div className="mb-8 p-4 bg-yellow-50 border border-yellow-200 rounded-xl text-yellow-800 text-sm font-medium">{dError}</div>
            )}

            {dResult && (
              <div>
                {/* Prediction change banner */}
                <div className={`mb-6 p-4 rounded-2xl border flex items-center gap-4 ${dResult.success ? 'bg-emerald-50 border-emerald-200' : 'bg-orange-50 border-orange-200'}`}>
                  <div className="text-3xl">{dResult.success ? '✅' : '⚠️'}</div>
                  <div>
                    <div className={`font-semibold text-base ${dResult.success ? 'text-emerald-800' : 'text-orange-800'}`}>
                      {dResult.success ? 'Defense Successful — Prediction Restored!' : 'Prediction Unchanged After Denoising'}
                    </div>
                    <div className="text-sm mt-0.5 text-gray-600">
                      Model was fooled into: <strong className="text-red-600">{dResult.original_prediction.label}</strong>
                      {' → '}Restored to: <strong className={dResult.success ? 'text-emerald-700' : 'text-orange-700'}>{dResult.restored_prediction.label}</strong>
                    </div>
                  </div>
                </div>

                {/* Side-by-side images */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                  <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100">
                    <div className="bg-red-50 aspect-square flex items-center justify-center p-4">
                      {dPreview && <img src={dPreview} className="max-w-full max-h-full object-contain rounded drop-shadow-md" />}
                    </div>
                    <div className="p-4 border-t border-gray-100">
                      <h4 className="text-xs uppercase text-red-400 font-semibold tracking-wider mb-1">Adversarial Input</h4>
                      <div className="font-semibold text-gray-900">{dResult.original_prediction.label}</div>
                      <div className="text-sm text-gray-500">{dResult.original_prediction.confidence.toFixed(2)}% confidence</div>
                    </div>
                  </div>

                  <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100 ring-2 ring-violet-500">
                    <div className="bg-violet-50 aspect-square flex items-center justify-center p-4">
                      <img src={dResult.denoised_image} className="max-w-full max-h-full object-contain rounded drop-shadow-md" />
                    </div>
                    <div className="p-4 border-t border-gray-100 bg-violet-50">
                      <h4 className="text-xs uppercase text-violet-400 font-semibold tracking-wider mb-1">Denoised Output</h4>
                      <div className="font-semibold text-violet-900">{dResult.restored_prediction.label}</div>
                      <div className="text-sm text-violet-600 mb-3">{dResult.restored_prediction.confidence.toFixed(2)}% confidence</div>
                      <button
                        onClick={() => downloadBase64Image(
                          dResult.denoised_image,
                          `denoised_${dMethod}.png`
                        )}
                        className="flex items-center justify-center gap-2 w-full py-2 bg-violet-600 hover:bg-violet-700 text-white rounded-lg text-sm font-semibold transition-all"
                      >
                        ↓ Download Denoised
                      </button>
                    </div>
                  </div>
                </div>

                {/* Metrics */}
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Image Quality Metrics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <MetricBadge label="PSNR" value={dResult.psnr_score} unit="dB" />
                  <MetricBadge label="SSIM" value={dResult.ssim_score} />
                  <MetricBadge label="Input Confidence" value={dResult.original_prediction.confidence} unit="%" />
                  <MetricBadge label="Restored Confidence" value={dResult.restored_prediction.confidence} unit="%" />
                </div>
                <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 text-xs text-gray-500 leading-relaxed">
                  <strong>PSNR</strong> (Peak Signal-to-Noise Ratio): higher is better. Above 30 dB indicates the denoised image is visually close to the input.{' '}
                  <strong>SSIM</strong> (Structural Similarity): ranges 0–1, values near 1 indicate high structural fidelity.
                </div>
              </div>
            )}
          </>
        )}
        {/* ════════════════════ DETECT TAB ════════════════════ */}
        {activeTab === 'detect' && (
          <>
            <div className="mb-6">
              <p className="text-gray-500 text-sm leading-relaxed max-w-2xl">
                Upload any image to check whether it contains adversarial perturbations.
                The detector analyzes gradient magnitude, FFT energy, Laplacian sharpness,
                and JPEG residual using the <strong>adv_ann_net</strong> architecture from the network module.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
              {/* Upload */}
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                <h2 className="text-lg font-semibold text-gray-800 mb-4">1. Upload Image to Inspect</h2>
                <label htmlFor="detect-file" className="flex flex-col items-center justify-center w-full h-36 border-2 border-dashed border-amber-300 rounded-xl cursor-pointer bg-amber-50 hover:bg-amber-100 transition-colors">
                  {dtPreview
                    ? <img src={dtPreview} className="h-28 w-auto object-contain rounded border border-amber-200" />
                    : <>
                        <span className="text-3xl mb-2">🔍</span>
                        <span className="text-sm text-amber-600 font-medium">Click to upload image for inspection</span>
                      </>}
                  <input id="detect-file" type="file" accept="image/*" className="hidden" onChange={e => {
                    if (e.target.files?.[0]) {
                      const f = e.target.files[0];
                      setDtFile(f); setDtPreview(URL.createObjectURL(f));
                      setDtResult(null); setDtError(null);
                    }
                  }} />
                </label>
              </div>

              {/* Config & Run */}
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex flex-col gap-5">
                <h2 className="text-lg font-semibold text-gray-800">2. Configure Detection</h2>
                <div>
                  <div className="flex justify-between items-end mb-1">
                    <label className="text-sm font-medium text-gray-700">Expected Epsilon <span className="text-xs text-gray-400">(attack bound to test against)</span></label>
                    <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">{dtEpsilon}</span>
                  </div>
                  <p className="text-xs text-gray-400 mb-2">Higher epsilon = larger attack signal = easier to detect.</p>
                  <input type="range" min="0.001" max="0.1" step="0.001" value={dtEpsilon}
                    onChange={e => setDtEpsilon(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500" />
                </div>
                <div className="bg-amber-50 border border-amber-100 rounded-xl p-4 text-xs text-amber-700 space-y-1">
                  <div className="font-semibold mb-1">Detection Signals Used:</div>
                  <div>📐 Gradient Magnitude — roughness of pixel transitions</div>
                  <div>📡 FFT High-Freq Energy — structured noise in frequency domain</div>
                  <div>🔲 Laplacian Sharpness — fine-grained edge noise</div>
                  <div>🗜️ JPEG Residual — compression artifact difference</div>
                </div>
                <button onClick={handleDetect} disabled={dtLoading || !dtFile}
                  className="mt-auto w-full bg-amber-500 hover:bg-amber-600 text-white font-medium py-3 px-4 rounded-xl shadow transition-colors disabled:bg-amber-200 flex items-center justify-center gap-2">
                  {dtLoading && <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/></svg>}
                  {dtLoading ? 'Analyzing...' : '🔍 Run Detection'}
                </button>
              </div>
            </div>

            {dtError && (
              <div className="mb-8 p-4 bg-yellow-50 border border-yellow-200 rounded-xl text-yellow-800 text-sm font-medium">{dtError}</div>
            )}

            {dtResult && (
              <div>
                {/* Verdict banner */}
                <div className={`mb-6 p-5 rounded-2xl border-2 flex items-center gap-5 ${
                  dtResult.is_adversarial
                    ? 'bg-red-50 border-red-300'
                    : 'bg-green-50 border-green-300'
                }`}>
                  <div className="text-5xl">{dtResult.is_adversarial ? '⚠️' : '✅'}</div>
                  <div className="flex-1">
                    <div className={`text-xl font-bold ${ dtResult.is_adversarial ? 'text-red-700' : 'text-green-700' }`}>
                      {dtResult.is_adversarial ? 'Adversarial Image Detected!' : 'Image Appears Clean'}
                    </div>
                    <div className="text-sm mt-1 text-gray-600">{dtResult.verdict}</div>
                    <div className="text-sm mt-1">
                      ResNet-18 classifies this as: <strong>{dtResult.prediction.label}</strong> ({dtResult.prediction.confidence.toFixed(1)}%)
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-gray-800">{(dtResult.confidence * 100).toFixed(1)}%</div>
                    <div className="text-xs text-gray-500">Detection confidence</div>
                    <div className="text-xs text-gray-500 mt-1">{dtResult.votes}/3 squeezers triggered</div>
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm mb-6">
                  <div className="flex justify-between text-sm font-medium text-gray-700 mb-2">
                    <span>Detection Confidence</span>
                    <span>{(dtResult.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-100 rounded-full h-4 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        dtResult.confidence > 0.75 ? 'bg-red-500' :
                        dtResult.confidence > 0.5  ? 'bg-amber-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${dtResult.confidence * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-400 mt-1">
                    <span>Clean</span><span>Adversarial</span>
                  </div>
                </div>

                <h3 className="text-lg font-semibold text-gray-800 mb-4">Squeezer Deltas <span className="text-sm text-gray-400 font-normal">(L1 softmax shift — larger = more adversarial)</span></h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  {([
                    { key: 'gaussian_blur_delta' as const,  label: '💨 Gaussian Blur',    icon: 'Blur' },
                    { key: 'bit_depth_delta'     as const,  label: '🖥️ Bit Depth',      icon: 'Bits' },
                    { key: 'jpeg_compress_delta' as const,  label: '🗜️ JPEG Compress',  icon: 'JPEG' },
                    { key: 'max_delta'           as const,  label: '📊 Max Delta',       icon: 'MAX'  },
                  ]).map(({ key, label }) => {
                    const val = dtResult.scores[key];
                    const thresh = dtResult.threshold;
                    const flagged = val > thresh;
                    return (
                      <div key={key} className={`rounded-xl p-4 border ${
                        flagged ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
                      }`}>
                        <div className="text-xs font-semibold mb-1 text-gray-600">{label}</div>
                        <div className={`text-lg font-bold font-mono ${ flagged ? 'text-red-600' : 'text-green-600' }`}>
                          {val.toFixed(4)}
                        </div>
                        <div className="text-xs mt-1 text-gray-400">threshold: {thresh.toFixed(4)}</div>
                        <div className={`text-xs font-bold mt-1 ${ flagged ? 'text-red-500' : 'text-green-500' }`}>
                          {flagged ? '🚩 Flagged' : '✓ Normal'}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Raw feature vector */}
                <details className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                  <summary className="text-sm font-semibold text-gray-600 cursor-pointer">Raw Feature Vector (12-D)</summary>
                  <div className="mt-3 grid grid-cols-4 md:grid-cols-6 gap-2">
                    {dtResult.features.map((v, i) => (
                      <div key={i} className="bg-white rounded-lg p-2 text-center border border-gray-100">
                        <div className="text-xs text-gray-400">f{i}</div>
                        <div className="text-xs font-mono text-gray-700">{v.toFixed(4)}</div>
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
