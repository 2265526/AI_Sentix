// API 封装：统一走 vite 代理（/v1 /rag /admin → 后端）
import axios from 'axios'

const http = axios.create({ timeout: 120000 })

// ---------- 聊天 /v1/chat/text ----------
export async function chatText(message, history = [], stream = false, sessionId) {
  const { data } = await http.post('/v1/chat/text', { message, history, stream, session_id: sessionId })
  return data
}

// 流式聊天（SSE）：sessionId 会话标识（短期记忆）；onMeta / onToken / onDone 回调
export async function chatTextStream(message, history = [], sessionId, { onMeta, onToken, onDone }) {
  const resp = await fetch('/v1/chat/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, stream: true, session_id: sessionId }),
  })
  if (!resp.ok || !resp.body) throw new Error(`聊天接口错误 ${resp.status}`)
  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    // 按 SSE 分隔符逐条解析
    const parts = buf.split('\n\n')
    buf = parts.pop()
    for (const part of parts) {
      const line = part.trim()
      if (!line.startsWith('data:')) continue
      try {
        const evt = JSON.parse(line.slice(5).trim())
        if (evt.type === 'meta') onMeta?.(evt)
        else if (evt.type === 'token') onToken?.(evt.content)
        else if (evt.type === 'done') onDone?.(evt)
      } catch (e) { /* 忽略无法解析的事件 */ }
    }
  }
  onDone?.({ type: 'done' })
}

// ---------- 阶段四：语音对话 /v1/chat/audio ----------
// 录音上传 → 返回 mp3 音频 Blob；识别文本/回复从响应头取（URL 编码）；
// sessionId 会话标识（短期记忆）；x-context-expired 表示后端会话过期
export async function chatAudio(file, history = [], sessionId) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('history', JSON.stringify(history))
  fd.append('session_id', sessionId)
  const resp = await fetch('/v1/chat/audio', { method: 'POST', body: fd })
  if (!resp.ok) {
    let detail = resp.statusText
    try { detail = (await resp.json()).detail || detail } catch (e) { /* ignore */ }
    throw new Error(detail)
  }
  const blob = await resp.blob()
  return {
    audioBlob: blob,
    transcript: decodeURIComponent(resp.headers.get('x-transcript') || ''),
    reply: decodeURIComponent(resp.headers.get('x-reply') || ''),
    intent: decodeURIComponent(resp.headers.get('x-intent') || ''),
    contextReset: resp.headers.get('x-context-expired') === 'true',
  }
}

// ---------- RAG 检索 /rag/search ----------
export async function ragSearch(query, top_k = 5, threshold = 0.4, doc_type = null) {
  const { data } = await http.post('/rag/search', { query, top_k, threshold, doc_type })
  return data
}

// ---------- 管理 /admin ----------
export async function uploadKb(file, docType) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('doc_type', docType)
  const { data } = await http.post('/admin/kb/upload', fd)
  return data
}

export async function importProducts(file) {
  const fd = new FormData()
  fd.append('file', file)
  const { data } = await http.post('/admin/products/import', fd)
  return data
}

export async function health() {
  const { data } = await http.get('/health')
  return data
}
