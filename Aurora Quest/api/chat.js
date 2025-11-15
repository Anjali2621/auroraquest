// api/chat.js
import fs from "fs";
import path from "path";

const STORE_PATH = path.resolve("./vectorstore.json");

// Load store
function loadStore() {
  try {
    const raw = fs.readFileSync(STORE_PATH, "utf8");
    return JSON.parse(raw);
  } catch (e) {
    return { docs: {}, chunks: [], df: {}, totalChunks: 0 };
  }
}

function tokenize(text) {
  const stop = new Set(["the","and","a","an","of","in","on","to","is","are","for","with","that","this","it","as","by","at","from","be","or","we","you"]);
  return text
    .toLowerCase()
    .replace(/[\r\n]+/g, " ")
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 2 && !stop.has(s));
}

function computeQueryTfIdf(tokens, store) {
  const tf = {};
  for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
  const maxTf = Math.max(...Object.values(tf), 1);
  const N = Math.max(1, store.totalChunks || store.chunks.length);
  const vec = {};
  for (const term in tf) {
    const tfNorm = tf[term] / maxTf;
    const df = store.df[term] || 0;
    const idf = Math.log((N) / (1 + df));
    vec[term] = tfNorm * idf;
  }
  return vec;
}

function computeChunkTfIdfVec(tfMap, store) {
  const N = Math.max(1, store.totalChunks || store.chunks.length);
  const vec = {};
  for (const term in tfMap) {
    const df = store.df[term] || 0;
    const idf = Math.log((N) / (1 + df));
    vec[term] = tfMap[term] * idf;
  }
  return vec;
}

function dotProduct(a, b) {
  let sum = 0;
  for (const k in a) if (b[k]) sum += a[k] * b[k];
  return sum;
}
function norm(a) {
  let s = 0;
  for (const k in a) s += a[k]*a[k];
  return Math.sqrt(s) + 1e-10;
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });
  try {
    const store = loadStore();
    if (!store || !store.chunks || store.chunks.length === 0) {
      return res.status(400).json({ error: "No documents indexed yet. Upload a PDF first." });
    }

    const { message, topK = 3, docs } = req.body || {};
    if (!message) return res.status(400).json({ error: "No message provided" });

    const qTokens = tokenize(message);
    if (qTokens.length === 0) return res.status(200).json({ answer: "Please ask using more keywords." });

    const qVec = computeQueryTfIdf(qTokens, store);

    // score all chunks (optionally filter by docs)
    const scored = [];
    for (const chunk of store.chunks) {
      if (Array.isArray(docs) && docs.length > 0 && !docs.includes(chunk.docId)) continue;
      const cVec = computeChunkTfIdfVec(chunk.tf || {}, store);
      const score = dotProduct(qVec, cVec) / (norm(qVec) * norm(cVec));
      scored.push({ score: isFinite(score) ? score : 0, text: chunk.text, docId: chunk.docId });
    }

    scored.sort((a,b)=>b.score - a.score);
    const top = scored.slice(0, topK).filter(s=>s.score>0);

    if (top.length === 0) {
      return res.json({ answer: "No relevant content found in uploaded materials." });
    }

    // Combine top chunks into a single answer (you can return separately too)
    const answer = top.map((t,i)=>`[Source ${i+1}] (doc: ${t.docId})\n\n${t.text}`).join("\n\n---\n\n");

    return res.json({ answer, rawScores: top.slice(0, topK) });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: e.message || "Server error" });
  }
}
