"use client";
import { useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [displayMessage, setDisplayMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) {
      setDisplayMessage("Please paste an article or claim to check for misinformation.");
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
          setDisplayMessage(`⚠️ Unexpected API response: ${data.result || JSON.stringify(data)}`);
          setResult("");
          setConfidence(null);
        }
      } else {
        setDisplayMessage(`⚠️ API Error: ${data.message || data.error || "An unknown error occurred from the backend."}`);
        setResult("");
        setConfidence(null);
      }
    } catch (err) {
      setDisplayMessage("🚫 Could not connect to the backend. Please ensure the server is running.");
      setResult("");
      setConfidence(null);
      console.error("API connection error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      className="min-h-screen flex flex-col items-center justify-start"
      style={{
        fontFamily: "Arial, sans-serif",
        background: "#f9f9f9",
        padding: "50px",
        textAlign: "center",
      }}
    >
      <h1 style={{ color: "#333", fontSize: "2rem", marginBottom: "1.5rem" }}>
        Misinformation Tracker
      </h1>

      <form onSubmit={handleSubmit} style={{ width: "100%", maxWidth: "700px" }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Paste an article or claim to check for misinformation"
          style={{
            width: "100%",
            height: "200px",
            padding: "10px",
            fontSize: "16px",
            resize: "vertical",
            overflowY: "auto",
            fontFamily: "inherit",
            border: "1px solid #ccc",
            borderRadius: "6px",
          }}
        />
        <br />
        <button
          type="submit"
          disabled={loading}
          style={{
            marginTop: "20px",
            padding: "10px 20px",
            fontSize: "16px",
            cursor: "pointer",
            backgroundColor: "#007bff",
            color: "#fff",
            border: "none",
            borderRadius: "4px",
            transition: "background 0.2s",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#0056b3")}
          onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#007bff")}
        >
          {loading ? (
            <span className="flex items-center gap-1">
              Classifying
              <span className="dot bounce1">●</span>
              <span className="dot bounce2">●</span>
              <span className="dot bounce3">●</span>
            </span>
          ) : (
            "Enter"
          )}
        </button>
      </form>

      <div id="result" style={{ marginTop: "30px", fontSize: "18px", color: "#555" }}>
        {displayMessage && <p>{displayMessage}</p>}
        {result && (
          <div
            style={{
              marginTop: "10px",
              backgroundColor: result.includes("✅")
                ? "#e6ffe6"
                : result.includes("❌")
                ? "#ffe6e6"
                : "#fff",
              color: result.includes("✅")
                ? "#28a745"
                : result.includes("❌")
                ? "#dc3545"
                : "#555",
              padding: "15px",
              borderRadius: "8px",
              fontSize: "18px",
              fontWeight: "500",
            }}
          >
            <p>{result}</p>
            {confidence !== null && (
              <p style={{ color: "#666", marginTop: "8px" }}>
                Confidence: {confidence}%
              </p>
            )}
          </div>
        )}
      </div>

      <div style={{ marginTop: "40px", color: "#666", fontSize: "16px" }}>
        <p>Developed by: Joshua, Ishrak, Swarali</p>
        <p>Guided by: Nathan</p>
      </div>

      {/* Animated Dots CSS */}
      <style jsx>{`
        .dot {
          font-size: 20px;
          line-height: 0;
          animation: bounce 1.4s infinite ease-in-out;
        }
        .bounce1 {
          animation-delay: 0s;
        }
        .bounce2 {
          animation-delay: 0.2s;
        }
        .bounce3 {
          animation-delay: 0.4s;
        }

        @keyframes bounce {
          0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.4;
          }
          40% {
            transform: scale(1);
            opacity: 1;
          }
        }
      `}</style>
    </main>
  );
}
