'use client';

import { useState, useEffect } from 'react';
import { Search, Mic, Brain, MapPin, X, Globe } from 'lucide-react';

// ✅ Simple sentiment detection (for positive English text)
// const isPositiveSentence = (text: string): boolean => {
//   const positives = [
//     "good", "great", "happy", "fine", "okay", "awesome", "excellent",
//     "nice", "positive", "better", "grateful", "joy", "love", "relaxed",
//   ];
//   const lowered = text.toLowerCase();
//   return positives.some(word => lowered.includes(word));
// };

export default function Home() {
  const [text, setText] = useState('');
  const [audio, setAudio] = useState<File | null>(null);
  const [eeg, setEeg] = useState<File | null>(null);
  const [fips, setFips] = useState('');
  const [predictions, setPredictions] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [languageMode, setLanguageMode] = useState<'english' | 'other'>('english');

  // Lightweight language check: English vs Non-English
  useEffect(() => {
    if (!text.trim()) {
      setLanguageMode('english');
      return;
    }

    const spanishAlphabetPattern = /[áéíóúüñÁÉÍÓÚÜÑ¡¿]/;
  const commonSpanishWords = /\b(el|la|de|que|y|en|a|los|las|me|te|porque|vida|feliz|jugar|gustar|uno|una|pero|por|para|todo|muy)\b/i;

    // If all characters are English letters, numbers, and punctuation → English
    const isEnglish = /^[A-Za-z0-9\s.,!?'"-]+$/.test(text) && !spanishAlphabetPattern.test(text) && !commonSpanishWords.test(text);
    setLanguageMode(isEnglish ? 'english' : 'other');
  }, [text]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text && !audio && !eeg && !fips) return;

    setLoading(true);
    setError('');
    setPredictions(null);

    // Skip API call for positive English text
    // if (languageMode === 'english' && isPositiveSentence(text)) {
    //   setPredictions({ text: 'Normal (No disorder detected)' });
    //   setLoading(false);
    //   return;
    // }

    const formData = new FormData();
    if (text) formData.append(languageMode === 'english' ? 'text' : 'multilingual_text', text);
    if (audio) formData.append('audio', audio);
    if (eeg) formData.append('eeg', eeg);
    if (fips) formData.append('fips_code', fips);

    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setPredictions(data.predictions);
    } catch (err: any) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const removeAudio = () => setAudio(null);
  const removeEeg = () => setEeg(null);
  const hasInput = text || audio || eeg || fips;

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center p-6">
      <div className="w-full max-w-3xl">
        <div className="backdrop-blur-xl bg-white/10 rounded-3xl shadow-2xl p-8 border border-white/20">
          <h1 className="text-4xl font-bold text-white text-center mb-8 tracking-tight">
         Multimodal Mental Disorder Prediction System
          </h1>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* 🌐 Smart Input */}
            <div className="relative">
              <input
                type="text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type your thoughts (Any Language)..."
                className="w-full px-6 py-5 pr-24 text-lg bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl text-white placeholder-white/60 focus:outline-none focus:ring-4 focus:ring-purple-500/50 transition-all"
              />
              <div className="absolute right-12 top-1/2 -translate-y-1/2 flex items-center gap-1">
                <Globe
                  className={`w-5 h-5 ${languageMode === 'english' ? 'text-green-400' : 'text-yellow-400'}`}
                />
                <span className="text-xs text-white/80">
                  {languageMode === 'english' ? 'EN' : 'Other'}
                </span>
              </div>
              <button
                type="submit"
                disabled={loading}
                className="absolute right-2 top-1/2 -translate-y-1/2 bg-gradient-to-r from-purple-600 to-pink-600 p-3 rounded-xl hover:scale-110 transition-transform"
              >
                <Search className="w-6 h-6 text-white" />
              </button>
            </div>

            {/*  File Uploads */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Audio */}
              <label className="cursor-pointer group">
                <input
                  type="file"
                  accept=".wav,.mp3"
                  onChange={(e) => setAudio(e.target.files?.[0] || null)}
                  className="hidden"
                />
                <div className="flex items-center justify-between p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20 hover:bg-white/20 transition-all group-hover:scale-105">
                  <div className="flex items-center gap-3">
                    <Mic className="w-6 h-6 text-pink-300" />
                    <div>
                      <p className="text-white font-medium">Audio</p>
                      <p className="text-white/60 text-xs truncate max-w-32">
                        {audio?.name || 'Voice Note'}
                      </p>
                    </div>
                  </div>
                  {audio && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.preventDefault();
                        removeAudio();
                      }}
                      className="text-red-400 hover:text-red-300"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  )}
                </div>
              </label>

              {/* EEG */}
              <label className="cursor-pointer group">
                <input
                  type="file"
                  accept=".mat"
                  onChange={(e) => setEeg(e.target.files?.[0] || null)}
                  className="hidden"
                />
                <div className="flex items-center justify-between p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20 hover:bg-white/20 transition-all group-hover:scale-105">
                  <div className="flex items-center gap-3">
                    <Brain className="w-6 h-6 text-cyan-300" />
                    <div>
                      <p className="text-white font-medium">EEG</p>
                      <p className="text-white/60 text-xs truncate max-w-32">
                        {eeg?.name || '.mat File'}
                      </p>
                    </div>
                  </div>
                  {eeg && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.preventDefault();
                        removeEeg();
                      }}
                      className="text-red-400 hover:text-red-300"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  )}
                </div>
              </label>

              {/* FIPS */}
              <div className="flex items-center gap-3 p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20">
                <MapPin className="w-6 h-6 text-green-300" />
                <input
                  type="text"
                  value={fips}
                  onChange={(e) => setFips(e.target.value)}
                  placeholder="FIPS Code"
                  autoComplete="off"
                  inputMode="numeric"
                  className="bg-transparent text-white placeholder-white/60 outline-none w-full"
                />
              </div>
            </div>

            {/* Loading Spinner */}
            {loading && (
              <div className="text-center">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-white border-t-transparent"></div>
                <p className="text-white mt-2">Analyzing...</p>
              </div>
            )}
          </form>

          {/* Results Section */}
          {predictions && hasInput && (
            <div className="mt-8 p-6 bg-white/10 backdrop-blur-md rounded-2xl border border-white/20">
              <h2 className="text-2xl font-bold text-white mb-4">AI Analysis</h2>
              <div className="space-y-4 text-white">
                {languageMode === 'english' && predictions.text && (
                  <div className="bg-gradient-to-r from-yellow-500/20 to-orange-500/20 p-4 rounded-xl border border-yellow-400/30">
                    <p className="text-sm opacity-80">Text Analysis (8 Disorders)</p>
                    <p className="text-2xl font-bold text-yellow-300">{predictions.text}</p>
                    {/* <p className="text-2xl font-bold text-yellow-300">{predictions.individual_predictions}</p> */}
                    
                  </div>
                )}

                {languageMode !== 'english' && predictions.multilingual && (
                  <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 p-4 rounded-xl border border-purple-400/30">
                    <p className="text-sm opacity-80">Multilingual Depression Screening</p>
                    <p className="text-2xl font-bold text-pink-300">{predictions.multilingual}</p>
                  </div>
                )}

                {predictions.audio && (
                  <div className="bg-white/10 p-4 rounded-xl border border-pink-400/30">
                    <p className="text-sm opacity-80">Voice Tone</p>
                    <p className="text-xl font-bold text-pink-300">{predictions.audio}</p>
                  </div>
                )}



                {predictions.eeg && (
                  <div className="bg-white/10 p-4 rounded-xl border border-cyan-400/30">
                    <p className="text-sm opacity-80">Brain Activity</p>
                    <p className="text-xl font-bold text-cyan-300">
                      {predictions.eeg}{' '}
                      {predictions.eeg_score > 0 && (
                        <span className="text-sm">
                          ({(predictions.eeg_score * 100).toFixed(1)}%)
                        </span>
                      )}
                    </p>
                  </div>
                )}

                {predictions.spatial_risk && (
                  <div className="bg-white/10 p-4 rounded-xl border border-cyan-400/30">
                    <h2 className='text-sm opacity-80'>Spatial Risk Prediction</h2>
                    <p><span className='text-xl font-bold text-cyan-300'>Risk:</span> {predictions.spatial_risk}</p>
                    <p><span className='text-xl font-bold text-cyan-300'>Score: </span> {predictions.spatial_score.toFixed(2)}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/*  Error */}
          {error && (
            <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-200 text-center">
              {error}
            </div>
          )}
        </div>

        <p className="text-center text-white/60 mt-6 text-sm">
        </p>
      </div>
    </div>
  );
}
