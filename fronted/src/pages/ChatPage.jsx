import React, { useRef, useState, useEffect } from 'react'
import { Button, Card, Input, Space, Typography, message, Spin } from 'antd'
import { SendOutlined, RobotOutlined, AudioOutlined } from '@ant-design/icons'
import { chatTextStream, chatAudio } from '../api.js'

const { TextArea } = Input

// 音量条数量（微信式：一排竖条随音量起伏）
const VOL_BARS = 15

// 消息：{ role: 'user'|'assistant', content, audioUrl? }
export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [recording, setRecording] = useState(false)
  const [volume, setVolume] = useState(0) // 0~1，录音实时音量
  const bottomRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const audioRef = useRef(null)
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const rafRef = useRef(null)

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
      // 平滑：音量过低压到 0，避免环境噪声撑起假音量
      setVolume(rms > 0.015 ? Math.min(rms * 2.2, 1) : 0)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }

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
    if (loading || recording) return  // 请求进行中/已在录音时禁止重复触发，避免并发上传
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
        stopMeter() // 结束录音：音量条消失
        const blob = new Blob(audioChunksRef.current, {
          type: mr.mimeType || 'audio/webm',
        })
        sendAudio(blob)
      }
      mr.start()
      startMeter(stream) // 开始录音：实时显示音量
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

  // 组件卸载时清理音量监测
  useEffect(() => () => stopMeter(), [])

  // 微信式音量条：一排竖条，随音量起伏（录音中显示，结束消失）
  const volumeBars = recording && (
    <div style={{ display: 'flex', alignItems: 'center', gap: 3, height: 24, marginBottom: 8, paddingLeft: 4 }}>
      {Array.from({ length: VOL_BARS }, (_, i) => {
        // 以中间为界向两侧起伏
        const active = i / VOL_BARS < volume
        const h = 6 + Math.round(volume * 16)
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
    <Card title="客服对话" style={{ width: '100%' }}>
      <div style={{ minHeight: 480, maxHeight: 620, overflowY: 'auto', padding: 8, marginBottom: 12 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: '#999', paddingTop: 140 }}>
            <RobotOutlined style={{ fontSize: 44 }} />
            <p style={{ fontSize: 16 }}>试试问：「iPhone 手机壳多少钱」「退货流程是怎样的」「连衣裙有货吗」</p>
            <p style={{ fontSize: 14 }}>🎤 点击下方麦克风按钮可直接语音提问（录音 → 识别 → 回复语音）</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 14 }}>
            <div style={{ maxWidth: '78%' }}>
              {m.role === 'user' ? (
                <div style={{ background: '#1677ff', color: '#fff', borderRadius: 10, padding: '10px 14px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 16, lineHeight: 1.7 }}>
                  {m.content}
                </div>
              ) : (
                <div style={{ background: '#fff', border: '1px solid #f0f0f0', borderRadius: 10, padding: '10px 14px', whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 16, lineHeight: 1.7 }}>
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
        <div ref={bottomRef} />
      </div>
      {volumeBars}
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
