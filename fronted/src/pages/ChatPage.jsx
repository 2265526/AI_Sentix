import React, { useRef, useState, useEffect } from 'react'
import { Button, Input, Typography, message, Spin } from 'antd'
import { RobotOutlined, AudioOutlined, PlusOutlined } from '@ant-design/icons'
import { chatTextStream, chatAudio } from '../api.js'

const { TextArea } = Input

// 音量条数量（微信式：一排竖条随音量起伏）
const VOL_BARS = 15

// 会话标识 localStorage 键：短期记忆按会话隔离，刷新/重开页面不丢
const SESSION_KEY = 'ai_sentix_session_id'

// 生成会话标识（优先 crypto.randomUUID；非安全上下文时降级随机串）
const genSessionId = () =>
  typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `sid-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

// 消息：{ role: 'user'|'assistant', content, audioUrl? }
export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [recording, setRecording] = useState(false)
  const [volume, setVolume] = useState(0) // 0~1，录音实时音量
  const scrollRef = useRef(null)          // 消息区滚动容器
  const nearBottomRef = useRef(true)      // 是否在底部范围（80px 内），初始视为在底部
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const audioRef = useRef(null)
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const rafRef = useRef(null)

  // ---------- 会话标识（短期记忆）：localStorage 持久化，刷新/重开不丢 ----------
  const [sessionId, setSessionId] = useState(() => {
    const saved = localStorage.getItem(SESSION_KEY)
    if (saved) return saved
    const sid = genSessionId()
    localStorage.setItem(SESSION_KEY, sid)
    return sid
  })
  const sessionIdRef = useRef(sessionId)

  const history = messages
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .slice(-10)
    .map((m) => ({ role: m.role, content: m.content }))

  // ---------- 录音音量监测（Web Audio 实时 RMS） ----------
  const stopMeter = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {})
      audioCtxRef.current = null
      analyserRef.current = null
    }
    setVolume(0)
  }

  const startMeter = (stream) => {
    const AudioCtx = window.AudioContext || window.webkitAudioContext
    if (!AudioCtx) return
    const ctx = new AudioCtx()
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 512
    ctx.createMediaStreamSource(stream).connect(analyser)
    audioCtxRef.current = ctx
    analyserRef.current = analyser
    const data = new Uint8Array(analyser.fftSize)
    const tick = () => {
      if (!analyserRef.current) return
      analyserRef.current.getByteTimeDomainData(data)
      let sum = 0
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128
        sum += v * v
      }
      const rms = Math.sqrt(sum / data.length)
      setVolume(rms > 0.015 ? Math.min(rms * 2.2, 1) : 0)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }

  // ---------- 智能跟随滚动 ----------
  // 在底部范围（80px 内）视为"在底部"：客服回复实时跟随下滑；
  // 用户向上翻阅历史则保持不动，滚回底部范围后恢复跟随。
  const BOTTOM_RANGE = 80

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    nearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_RANGE
  }

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.addEventListener('scroll', handleScroll, { passive: true })
    return () => el?.removeEventListener('scroll', handleScroll)
  }, [])

  // 消息更新（含流式 token）时：仅在底部范围才跟随滚动，否则保持用户阅读位置
  useEffect(() => {
    const el = scrollRef.current
    if (el && nearBottomRef.current) el.scrollTop = el.scrollHeight
  }, [messages])

  // ---------- 新建会话：重新生成 session_id 覆盖 localStorage，清空消息历史与状态 ----------
  const resetSession = () => {
    const sid = genSessionId()
    localStorage.setItem(SESSION_KEY, sid)
    sessionIdRef.current = sid
    setSessionId(sid)
    setMessages([])
    setInput('')
    setLoading(false)
    setStreaming(false)
  }

  // ---------- 文本发送（回车即发，无发送按钮） ----------
  const send = async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    const userMsg = { role: 'user', content: text }
    setMessages((ms) => [...ms, userMsg])
    setLoading(true)
    setStreaming(false)
    const sid = sessionIdRef.current // 固定本次请求的会话标识
    try {
      await chatTextStream(text, history, sid, {
        onMeta: (evt) => {
          if (sid !== sessionIdRef.current) return // 已新建会话，忽略旧流回调
          // 后端短期记忆过期：本地有历史时清空界面并温柔提示，随后开启新对话
          if (evt.context_reset === true && messages.length > 0) {
            setMessages([])
            message.info('会话已过期，已开启新对话')
          }
          setStreaming(true)
          setMessages((ms) => [...ms, { role: 'assistant', content: '' }])
        },
        onToken: (delta) => {
          if (sid !== sessionIdRef.current) return
          setMessages((ms) => {
            if (ms.length === 0) return ms
            const last = ms[ms.length - 1]
            if (last?.role !== 'assistant') return ms
            return [...ms.slice(0, -1), { ...last, content: last.content + delta }]
          })
        },
        onDone: () => {
          if (sid !== sessionIdRef.current) return
          setStreaming(false)
          setLoading(false)
        },
      })
    } catch (e) {
      message.error(`请求失败：${e.message}`)
      setLoading(false)
      setStreaming(false)
    }
  }

  // ---------- 语音对话（录音 → /v1/chat/audio → 播放回复） ----------
  // 麦克风状态独立：只有用户点击录音（recording）才显示"使用中"，
  // 客服回复（LLM 处理 / 播放语音）不影响麦克风状态，可与用户录音并发
  const startRecord = async () => {
    if (recording) return  // 仅防连点重复录音；请求进行中不拦截（麦克风与客服回复并发）
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
        stopMeter()
        const blob = new Blob(audioChunksRef.current, {
          type: mr.mimeType || 'audio/webm',
        })
        sendAudio(blob)
      }
      mr.start()
      startMeter(stream)
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
      const r = await chatAudio(blob, history, sessionIdRef.current)
      // 语音会话过期：本地有历史时清空并提示，然后展示本轮识别与回复
      if (r.contextReset && messages.length > 0) {
        setMessages([])
        message.info('会话已过期，已开启新对话')
      }
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
  }

  useEffect(() => () => stopMeter(), [])

  // 微信式音量条（录音中显示在输入区上方）
  const volumeBars = recording && (
    <div style={{ display: 'flex', alignItems: 'center', gap: 3, height: 22, padding: '0 4px' }}>
      {Array.from({ length: VOL_BARS }, (_, i) => {
        const active = i / VOL_BARS < volume
        const h = 6 + Math.round(volume * 15)
        return (
          <div
            key={i}
            style={{
              width: 4,
              height: h,
              borderRadius: 2,
              background: active ? '#1677ff' : '#d9d9d9',
              transition: 'height 80ms linear',
            }}
          />
        )
      })}
      <Typography.Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>
        {volume > 0.01 ? '正在聆听…' : '说话吧，麦克风已开启'}
      </Typography.Text>
    </div>
  )

  return (
    <div
      style={{
        height: 'calc(100vh - 56px)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'stretch',
        background: '#f5f5f5',
      }}
    >
      {/* 聊天面板：水平居中，宽为视口的 2/3，纵向撑满（消息区最大化） */}
      <div
        style={{
          width: '66.67%',
          minWidth: 520,
          display: 'flex',
          flexDirection: 'column',
          background: '#f5f5f5',
        }}
      >
        {/* 消息区（相对定位，圆形麦克风按钮浮动右下角） */}
        <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', position: 'relative' }}>
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', color: '#999', paddingTop: '20vh' }}>
              <RobotOutlined style={{ fontSize: 44 }} />
              <p style={{ fontSize: 16, marginTop: 12 }}>试试问：「iPhone 手机壳多少钱」「退货流程是怎样的」「连衣裙有货吗」</p>
              <p style={{ fontSize: 14 }}>🎤 点击右下角麦克风可直接语音提问（录音 → 识别 → 回复语音）</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 16 }}>
              <div style={{ maxWidth: '76%' }}>
                {m.role === 'user' ? (
                  <div style={{ background: '#1677ff', color: '#fff', borderRadius: 12, padding: '10px 14px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 16, lineHeight: 1.7 }}>
                    {m.content}
                  </div>
                ) : (
                  <div style={{ background: '#fff', borderRadius: 12, padding: '10px 14px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 16, lineHeight: 1.7, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
                    {m.content || (streaming ? <Spin size="small" /> : '')}
                    {m.audioUrl && (
                      <div style={{ marginTop: 10 }}>
                        <audio ref={audioRef} controls src={m.audioUrl} style={{ width: '100%', height: 36 }} />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* 音量条 */}
        {volumeBars}

        {/* 输入区：独立卡片（白底圆角，与聊天面板分离），回车即发送；
            麦克风按钮固定在输入框右下角，大小适配输入框 */}
        <div style={{ padding: '8px 24px 16px' }}>
          {/* 新建会话：生成新 session_id 并清空当前对话（短期记忆重新开始） */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
            <Button size="small" icon={<PlusOutlined />} onClick={resetSession}>
              新建会话
            </Button>
          </div>
          <div
            style={{
              background: '#fff',
              border: '1px solid #e8e8e8',
              borderRadius: 12,
              boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
              padding: '8px 10px',
              display: 'flex',
              alignItems: 'flex-end',
              gap: 6,
            }}
          >
            <TextArea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="输入你的问题，回车发送…"
              autoSize={{ minRows: 3, maxRows: 6 }}
              onPressEnter={(e) => { if (!e.shiftKey) { e.preventDefault(); send() } }}
              disabled={loading}
              variant="borderless"
              style={{ flex: 1, fontSize: 16, lineHeight: 1.6, resize: 'none', padding: 0 }}
            />
            <Button
              shape="circle"
              icon={<AudioOutlined style={{ fontSize: 20 }} />}
              onClick={recording ? stopRecord : startRecord}
              danger={recording}
              style={{
                width: 44,
                height: 44,
                flexShrink: 0,
                background: recording ? '#ff4d4f' : '#1677ff',
                borderColor: recording ? '#ff4d4f' : '#1677ff',
                color: '#fff',
                boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
              }}
            />
          </div>
          <Typography.Text type="secondary" style={{ fontSize: 12, marginTop: 6, display: 'block', textAlign: 'right' }}>
            回车发送 · Shift+Enter 换行 · 支持语音提问
          </Typography.Text>
        </div>
      </div>
    </div>
  )
}
