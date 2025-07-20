"use client";
import { useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [displayMessage, setDisplayMessage] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!input.trim()) {
      setDisplayMessage(
        "Please paste an article or claim to check for misinformation."
      );
      setResult("");
      setConfidence(null);
      return;
    }

    setLoading(true);
    setResult("");
    setConfidence(null);

    try {
      const res = await fetch("http://localhost:5050/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ news_input: input }),
      });

      const data = await res.json();

      if (res.ok) {
        if (data.result === "🟩 Real News") {
          setDisplayMessage(null);
          setResult("✅ This claim appears to be reliable.");
          setConfidence(data.confidence);
        } else if (data.result === "🟥 Fake News") {
          setDisplayMessage(null);
          setResult("❌ This is classified as misinformation.");
          setConfidence(data.confidence);
        } else {
          setDisplayMessage(
            `⚠️ Unexpected API response: ${data.result || JSON.stringify(data)}`
          );
          setResult("");
          setConfidence(null);
        }
      } else {
        setDisplayMessage(
          `⚠️ API Error: ${
            data.message ||
            data.error ||
            "An unknown error occurred from the backend."
          }`
        );
        setResult("");
        setConfidence(null);
      }
    } catch (err) {
      setDisplayMessage(
        "🚫 Could not connect to the backend. Please ensure the server is running."
      );
      setResult("");
      setConfidence(null);
      console.error("API connection error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-start bg-gray-50 px-12 py-12 text-center">
      <h1 className="text-gray-800 text-3xl mb-6 font-sans">
        Misinformation Tracker
      </h1>

      <div className="w-full max-w-2xl">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Paste an article or claim to check for misinformation"
          className="w-full h-48 p-3 text-base text-black placeholder-gray-500 resize-y overflow-y-auto font-sans border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="mt-5 px-5 py-2.5 text-base cursor-pointer bg-blue-600 text-white border-none rounded inline-flex items-center justify-center transition-colors duration-200 hover:bg-blue-700 disabled:cursor-not-allowed"
        >
          {loading ? (
            <span className="flex items-center gap-1">
              Classifying
              <span className="text-xl leading-none animate-bounce delay-0">
                ●
              </span>
              <span className="text-xl leading-none animate-bounce delay-200">
                ●
              </span>
              <span className="text-xl leading-none animate-bounce delay-400">
                ●
              </span>
            </span>
          ) : (
            "Enter"
          )}
        </button>
      </div>

      <div className="mt-8 text-lg text-gray-600">
        {displayMessage && <p>{displayMessage}</p>}
        {result && (
          <div
            className={`mt-2.5 p-4 rounded-lg text-lg font-medium ${
              result.includes("✅")
                ? "bg-green-50 text-green-600"
                : result.includes("❌")
                ? "bg-red-50 text-red-600"
                : "bg-white text-gray-600"
            }`}
          >
            <p>{result}</p>
            {confidence !== null && (
              <p className="text-gray-500 mt-2">Confidence: {confidence}%</p>
            )}
          </div>
        )}
      </div>

      <div className="mt-10 text-gray-500 text-base">
        <p>Developed by: Joshua, Ishrak, Swarali</p>
        <p>Guided by: Nathan</p>
      </div>

      <style jsx>{`
        .delay-0 {
          animation-delay: 0s;
        }
        .delay-200 {
          animation-delay: 0.2s;
        }
        .delay-400 {
          animation-delay: 0.4s;
        }
      `}</style>
    </main>
  );
}
