// api/upload.js
import formidable from "formidable";
import fs from "fs";
import path from "path";
import pdfParse from "pdf-parse";
import mammoth from "mammoth";
import { v4 as uuidv4 } from "uuid";

export const config = {
  api: { bodyParser: false }
};

const STORE_PATH = path.resolve("./vectorstore.json");
const CHUNK_MAX_CHARS = 1200; // chunk size (approx characters)

function loadStore() {
  try {
    const raw = fs.readFileSync(STORE_PATH, "utf8");
    return JSON.parse(raw);
  } catch (e) {
    return { docs: {}, chunks: [], df: {}, totalChunks: 0 };
  }
}
function saveStore(store) {
  fs.writeFileSync(STORE_PATH, JSON.stringify(store, null, 2));
}

function tokenize(text) {
  // simple tokenizer + lower + remove non-word, remove short tokens and basic stopwords
  const stop = new Set(["the","and","a","an","of","in","on","to","is","are","for","with","that","this","it","as","by","at","from","be","or","we","you"]);
  return text
    .toLowerCase()
    .replace(/[\r\n]+/g, " ")
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 2 && !stop.has(s));
}

function chunkText(text, maxChars=CHUNK_MAX_CHARS) {
  const paras = text.split(/\n{1,}/).map(p=>p.trim()).filter(Boolean);
  const chunks = [];
  let buf = "";
  for (const p of paras) {
    if ((buf + " " + p).length > maxChars) {
      if (buf) chunks.push(buf.trim());
      buf = p;
      // if single paragraph too long, split it
      while (buf.length > maxChars) {
        chunks.push(buf.slice(0, maxChars).trim());
        buf = buf.slice(maxChars);
      }
    } else {
      buf = buf ? (buf + "\n\n" + p) : p;
    }
  }
  if (buf) chunks.push(buf.trim());
  return chunks;
}

async function fileToText(file) {
  const buf = await fs.promises.readFile(file.filepath ?? file.path);
  const name = file.originalFilename ?? file.name;
  if (name.toLowerCase().endsWith(".pdf") || file.mimetype === "application/pdf") {
    const data = await pdfParse(buf);
    return data.text || "";
  } else if (name.toLowerCase().endsWith(".docx") || file.mimetype?.includes("word")) {
    const res = await mammoth.extractRawText({ buffer: buf });
    return res.value || "";
  } else {
    return buf.toString("utf8");
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const form = formidable({ multiples: false, keepExtensions: true });
  form.parse(req, async (err, fields, files) => {
    if (err) return res.status(500).json({ error: "Form parse error" });
    const file = files.file;
    if (!file) return res.status(400).json({ error: "No file uploaded (field name must be `file`)" });

    try {
      const text = await fileToText(file);
      if (!text || text.trim().length === 0) return res.status(400).json({ error: "No text found in file" });

      const store = loadStore();
      const docId = uuidv4();
      const chunks = chunkText(text, CHUNK_MAX_CHARS);

      // For each chunk compute term frequencies and update document frequency (df)
      for (let i = 0; i < chunks.length; i++) {
        const chunkText = chunks[i];
        const tokens = tokenize(chunkText);
        const tf = {};
        const seenTerms = new Set();
        for (const t of tokens) {
          tf[t] = (tf[t] || 0) + 1;
          seenTerms.add(t);
        }
        // normalize tf by chunk length (optional)
        const maxTf = Math.max(...Object.values(tf), 1);
        for (const k in tf) tf[k] = tf[k] / maxTf;

        // update DF counts
        for (const term of seenTerms) {
          store.df[term] = (store.df[term] || 0) + 1;
        }

        const chunkId = uuidv4();
        store.chunks.push({
          id: chunkId,
          docId,
          text: chunkText,
          tf // store TF map for quick TF-IDF later
        });
        store.totalChunks = (store.totalChunks || 0) + 1;
      }

      store.docs[docId] = {
        id: docId,
        name: file.originalFilename ?? file.name,
        chunkCount: chunks.length,
        uploadedAt: new Date().toISOString()
      };

      saveStore(store);

      return res.json({ id: docId, name: store.docs[docId].name, chunks: chunks.length });
    } catch (e) {
      console.error(e);
      return res.status(500).json({ error: e.message || "Upload processing failed" });
    }
  });
}
