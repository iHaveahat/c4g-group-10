"use client";
import { useState } from "react";
import Image from "next/image";





export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setLoading(true);
    setResult("");
    setConfidence(null);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ news_input: input }),
      });

      const data = await res.json();

      if (res.ok) {
        setResult(data.result);
        setConfidence(data.confidence);
      } else {
        setResult(data.message || "Error from backend.");
      }
    } catch (err) {
      setResult("Could not connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold mb-6">Fake News Classifier</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Paste a news article or link..."
          className="w-full p-3 border border-gray-300 rounded-lg resize-none"
          rows={6}
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-5 py-2 rounded hover:bg-blue-700 transition"
        >
          {loading ? "Classifying..." : "Submit"}
        </button>
      </form>

      {result && (
        <div className="mt-6 p-4 bg-gray-100 rounded-lg shadow-sm">
          <p className="text-xl">{result}</p>
          {confidence !== null && <p className="text-gray-700">Confidence: {confidence}%</p>}
        </div>
      )}
    </main>
  );
}
