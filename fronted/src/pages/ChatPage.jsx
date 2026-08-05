import React, { useRef, useState } from 'react'
import { Button, Card, Input, Space, Typography, message, Spin } from 'antd'
import { SendOutlined, RobotOutlined } from '@ant-design/icons'
import { chatTextStream } from '../api.js'

const { TextArea } = Input

// 消息：{ role: 'user'|'assistant', content }
export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef(null)

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

  return (
    <Card title="客服对话" style={{ maxWidth: 900, margin: '0 auto' }}>
      <div style={{ minHeight: 420, maxHeight: 560, overflowY: 'auto', padding: 8, marginBottom: 12 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: '#999', paddingTop: 120 }}>
            <RobotOutlined style={{ fontSize: 40 }} />
            <p>试试问：「iPhone 手机壳多少钱」「退货流程是怎样的」「连衣裙有货吗」</p>
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
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <Space.Compact style={{ width: '100%' }}>
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
        说明：未登录验证台；回复基于工具检索结果（价格/库存/RAG 知识）生成，流式展示。
      </Typography.Text>
    </Card>
  )
}
