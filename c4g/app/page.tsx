"use client";
import { useState } from "react";
// import Image from "next/image"; // Not used in this component, can be removed

export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [displayMessage, setDisplayMessage] = useState<string | null>(null); // State for user-facing messages

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
    setDisplayMessage("⚙️ Analyzing content... Please wait."); // Show loading message

    try {
      const res = await fetch("http://localhost:5050/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ news_input: input }),
      });

      const data = await res.json();

      if (res.ok) {
        // Based on Flask API's expected output (🟩 Real News / 🟥 Fake News)
        if (data.result === "🟩 Real News") {
          setDisplayMessage(null); // Clear loading/error message
          setResult(`✅ This claim appears to be reliable.`);
          setConfidence(data.confidence);
        } else if (data.result === "🟥 Fake News") {
          setDisplayMessage(null); // Clear loading/error message
          setResult(`❌ This is classified as misinformation.`);
          setConfidence(data.confidence);
        } else {
          // Fallback for unexpected 'data.result' values
          setDisplayMessage(`⚠️ Unexpected API response: ${data.result || JSON.stringify(data)}`);
          setResult(""); // Clear result
          setConfidence(null);
        }
      } else {
        // Handle API errors (e.g., 400 Bad Request, 500 Internal Server Error)
        setDisplayMessage(`⚠️ API Error: ${data.message || data.error || "An unknown error occurred from the backend."}`);
        setResult("");
        setConfidence(null);
      }
    } catch (err) {
      // Handle network errors or issues
      setDisplayMessage("🚫 Could not connect to the backend. Please ensure the server is running.");
      setResult("");
      setConfidence(null);
      console.error("API connection error:", err); // Log full error for debugging
    } finally {
      setLoading(false); // stop loading animation
    }
  };

  return (
    <main className="max-w-xl mx-auto px-4 py-10" // main layout classes
            style={{ 
                fontFamily: 'Arial, sans-serif', 
                textAlign: 'center', 
                background: '#f9f9f9', 
                padding: '50px' 
            }}
    >
      {/* H1 */}
      <h1 className="text-3xl font-bold mb-6 text-black">Misinformation Tracker</h1> {/* text-gray-800 approximates #333 */}

      <form onSubmit={handleSubmit} className="space-y-4">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Paste an article or claim to check for misinformation"
          className="w-full p-3 border border-gray-300 rounded-lg resize-y overflow-auto text-base" 
          rows={6} 
          style={{ height: '200px' }} 
        />
        <button
          type="submit"
          className="mt-5 px-5 py-2 text-base cursor-pointer bg-blue-600 text-white rounded hover:bg-blue-700 transition" 
          disabled={loading} // loading disabled state
        >
          {loading ? "Classifying..." : "Enter"}
        </button>
      </form>

      {/* Result/Message Display Area */}
      <div id="result-display" className="mt-8 text-lg text-black"> {/* Adjusted margin-top*/}
        {/* Loading/Error/Initial messages */}
        {displayMessage && (
            <p className={`${loading ? 'text-gray-600' : 'text-yellow-600'}`}> {/* Gray for loading, yellow for warnings */}
                {displayMessage}
            </p>
        )}

        {/* Prediction Result */}
        {result && (
          <div className="mt-4 p-4 rounded-lg shadow-sm"
               style={{ 
                   backgroundColor: result.includes('✅') ? '#e6ffe6' : (result.includes('❌') ? '#ffe6e6' : '#fff'), // Light green/red/white background based on result
                   color: result.includes('✅') ? '#28a745' : (result.includes('❌') ? '#dc3545' : '#555'), // Dark green/red/default text color
                   fontSize: '18px',
                   textAlign: 'center' // Ensure text aligns center
               }}
          >
            <p className="text-xl font-semibold">{result}</p>
            {confidence !== null && <p className="text-gray-700">Confidence: {confidence}%</p>}
          </div>
        )}
      </div>

      <div className="mt-10 pt-5 text-gray-600"> {/* Increased margin-top for more separation */}
        <p>Developed by: Joshua, Ishrak, Swarali</p>
        <p>Guided by: Nathan</p>
      </div>
    </main>
  );
}
