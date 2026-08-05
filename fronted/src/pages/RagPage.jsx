import React, { useState } from 'react'
import {
  Button, Card, Col, Descriptions, Input, Row, Select, Space, Tag, Typography, message,
} from 'antd'
import { SearchOutlined } from '@ant-design/icons'
import { ragSearch } from '../api.js'

export default function RagPage() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [threshold, setThreshold] = useState(0.4)
  const [docType, setDocType] = useState(undefined)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const search = async () => {
    if (!query.trim()) return message.warning('请输入检索问题')
    setLoading(true)
    try {
      const data = await ragSearch(query, topK, threshold, docType || null)
      setResult(data)
    } catch (e) {
      message.error(`检索失败：${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto' }}>
      <Card title="RAG 检索验证（阶段二：BM25 + 向量双路召回 → Rerank 阈值过滤 → Top-K）">
        <Space direction="vertical" style={{ width: '100%' }} size={12}>
          <Space.Compact style={{ width: '100%' }}>
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="输入检索问题，如：退货流程是怎样的 / 手机壳"
              onPressEnter={search}
              size="large"
            />
            <Button type="primary" icon={<SearchOutlined />} onClick={search} loading={loading} size="large">
              检索
            </Button>
          </Space.Compact>
          <Space wrap>
            <span>Top-K：
              <Select value={topK} onChange={setTopK} style={{ width: 70 }}
                options={[3, 5, 10].map((v) => ({ value: v, label: v }))} />
            </span>
            <span>阈值：
              <Select value={threshold} onChange={setThreshold} style={{ width: 90 }}
                options={[0.3, 0.4, 0.5, 0.6].map((v) => ({ value: v, label: String(v) }))} />
            </span>
            <span>文档类型：
              <Select value={docType} onChange={setDocType} style={{ width: 150 }} allowClear
                placeholder="全部"
                options={[
                  { value: 'product_manual', label: '商品说明书' },
                  { value: 'faq', label: '常见问题 FAQ' },
                  { value: 'policy', label: '政策' },
                ]} />
            </span>
          </Space>
        </Space>
      </Card>

      {result && (
        <Card style={{ marginTop: 16 }} title={
          <Space>
            <span>检索结果：{result.total} 条</span>
            <Tag color={result.degraded ? 'orange' : 'green'}>
              {result.degraded ? '降级（仅 BM25）' : '混合检索正常'}
            </Tag>
          </Space>
        }>
          <Descriptions size="small" column={3} style={{ marginBottom: 12 }}>
            <Descriptions.Item label="Query">{result.query}</Descriptions.Item>
            <Descriptions.Item label="阈值">{result.threshold}</Descriptions.Item>
            <Descriptions.Item label="总数">{result.total}</Descriptions.Item>
          </Descriptions>
          {result.results.length === 0 && <Typography.Text type="secondary">没有通过阈值的结果。</Typography.Text>}
          {result.results.map((r, i) => (
            <Card key={r.chunk_id} size="small" style={{ marginBottom: 8 }}>
              <Space wrap style={{ marginBottom: 6 }}>
                <Tag color="blue">#{i + 1}</Tag>
                <Tag>{r.doc_type || 'unknown'}</Tag>
                <Tag color="gold">综合分 {r.score}</Tag>
                {r.vector_score != null && <Tag color="cyan">向量 {r.vector_score}</Tag>}
                {r.bm25_score != null && <Tag color="purple">BM25 {r.bm25_score}</Tag>}
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  chunk_id={r.chunk_id} doc_id={r.doc_id} idx={r.chunk_index}
                </Typography.Text>
              </Space>
              <Typography.Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
                {r.chunk_text}
              </Typography.Paragraph>
            </Card>
          ))}
        </Card>
      )}
    </div>
  )
}
