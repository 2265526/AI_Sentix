import React, { useRef, useState } from 'react'
import { Button, Card, Input, Space, Typography, message, Spin } from 'antd'
import { SendOutlined, RobotOutlined, AudioOutlined } from '@ant-design/icons'
import { chatTextStream, chatAudio } from '../api.js'

const { TextArea } = Input

// 消息：{ role: 'user'|'assistant', content, audioUrl? }
export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [recording, setRecording] = useState(false)
  const bottomRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const audioRef = useRef(null)

  const history = messages
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .slice(-10)
    .map((m) => ({ role: m.role, content: m.content }))

  const send = async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    const userMsg = { role: 'user', content: text }
    setMessages((ms) => [...ms, userMsg])
    setLoading(true)
    setStreaming(false)
    try {
      await chatTextStream(text, history, {
        onMeta: () => {
          setStreaming(true)
          setMessages((ms) => [...ms, { role: 'assistant', content: '' }])
        },
        onToken: (delta) => {
          setMessages((ms) => {
            if (ms.length === 0) return ms
            const last = ms[ms.length - 1]
            if (last?.role !== 'assistant') return ms
            // 不可变更新（纯函数）：StrictMode 已移除，但保持幂等，双调用不重复
            return [...ms.slice(0, -1), { ...last, content: last.content + delta }]
          })
        },
        onDone: () => {
          setStreaming(false)
          setLoading(false)
        },
      })
    } catch (e) {
      message.error(`请求失败：${e.message}`)
      setLoading(false)
      setStreaming(false)
    }
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
  }

  // ---------- 阶段四：语音对话（录音 → /v1/chat/audio → 播放回复） ----------
  const startRecord = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      message.warning('当前浏览器不支持录音（需 HTTPS 或 localhost）')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream)
      mediaRecorderRef.current = mr
      audioChunksRef.current = []
      mr.ondataavailable = (e) => audioChunksRef.current.push(e.data)
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(audioChunksRef.current, {
          type: mr.mimeType || 'audio/webm',
        })
        sendAudio(blob)
      }
      mr.start()
      setRecording(true)
    } catch (e) {
      message.error(`无法获取麦克风：${e.message}`)
    }
  }

  const stopRecord = () => {
    mediaRecorderRef.current?.stop()
    setRecording(false)
  }

  const sendAudio = async (blob) => {
    setLoading(true)
    try {
      const r = await chatAudio(blob, history)
      setMessages((ms) => [...ms, { role: 'user', content: `🎤 ${r.transcript || '（未识别到语音）'}` }])
      if (r.reply) {
        const audioUrl = URL.createObjectURL(r.audioBlob)
        setMessages((ms) => [...ms, { role: 'assistant', content: r.reply, audioUrl }])
        setTimeout(() => audioRef.current?.play(), 150)
      } else {
        message.warning('语音合成失败，已返回文字回复')
      }
    } catch (e) {
      message.error(`语音请求失败：${e.message}`)
    } finally {
      setLoading(false)
    }
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
  }

  return (
    <Card title="客服对话" style={{ maxWidth: 900, margin: '0 auto' }}>
      <div style={{ minHeight: 420, maxHeight: 560, overflowY: 'auto', padding: 8, marginBottom: 12 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: '#999', paddingTop: 120 }}>
            <RobotOutlined style={{ fontSize: 40 }} />
            <p>试试问：「iPhone 手机壳多少钱」「退货流程是怎样的」「连衣裙有货吗」</p>
            <p style={{ fontSize: 13 }}>🎤 点击下方麦克风按钮可直接语音提问（录音 → 识别 → 回复语音）</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 12 }}>
            <div style={{ maxWidth: '78%' }}>
              {m.role === 'user' ? (
                <div style={{ background: '#1677ff', color: '#fff', borderRadius: 10, padding: '8px 12px', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {m.content}
                </div>
              ) : (
                <div style={{ background: '#fff', border: '1px solid #f0f0f0', borderRadius: 10, padding: '8px 12px', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {m.content || (streaming ? <Spin size="small" /> : '')}
                  {m.audioUrl && (
                    <div style={{ marginTop: 8 }}>
                      <audio ref={audioRef} controls src={m.audioUrl} style={{ width: '100%', height: 34 }} />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <Space.Compact style={{ width: '100%' }}>
        <Button
          icon={<AudioOutlined />}
          onClick={recording ? stopRecord : startRecord}
          loading={loading}
          danger={recording}
          style={{ height: 'auto' }}
        >
          {recording ? '停止录音' : '语音提问'}
        </Button>
        <TextArea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="输入你的问题…"
          autoSize={{ minRows: 1, maxRows: 3 }}
          onPressEnter={(e) => { if (!e.shiftKey) { e.preventDefault(); send() } }}
          disabled={loading}
        />
        <Button type="primary" icon={<SendOutlined />} onClick={send} loading={loading} style={{ height: 'auto' }}>
          发送
        </Button>
      </Space.Compact>
      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
        说明：未登录验证台；支持文字与语音提问（录音→ASR→LLM→TTS，首次语音请求需加载识别模型约 1 分钟）。
      </Typography.Text>
    </Card>
  )
}
