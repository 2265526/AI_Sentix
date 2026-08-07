import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Drawer,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Timeline,
  Typography,
} from 'antd'
import { ReloadOutlined, SyncOutlined } from '@ant-design/icons'
import { getMonitorRequest, getMonitorRequests, getMonitorSummary } from '../api.js'

const { Text } = Typography

// 状态 → 颜色/文案
const STATUS_COLOR = { ok: 'green', degraded: 'orange', error: 'red', skipped: 'default' }

export default function MonitorPage() {
  const [summary, setSummary] = useState(null)
  const [items, setItems] = useState([])
  const [status, setStatus] = useState('all')
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const timerRef = useRef(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [s, r] = await Promise.all([getMonitorSummary(), getMonitorRequests({ limit: 100, status })])
      setSummary(s)
      setItems(r.items || [])
    } catch (e) {
      console.error('监控数据加载失败:', e)
    } finally {
      setLoading(false)
    }
  }, [status])

  // 首次加载 + 状态切换时加载
  useEffect(() => { load() }, [load])

  // 自动刷新（每 5s）
  useEffect(() => {
    if (autoRefresh) {
      timerRef.current = setInterval(load, 5000)
      return () => clearInterval(timerRef.current)
    }
  }, [autoRefresh, load])

  const openDetail = useCallback(async (id) => {
    try {
      setDetail(await getMonitorRequest(id))
    } catch (e) {
      console.error('详情加载失败:', e)
    }
  }, [])

  const columns = [
    { title: '时间', dataIndex: 'ts', width: 150 },
    { title: '问题', dataIndex: 'query', ellipsis: true, width: 260,
      render: (v) => <Text style={{ fontSize: 12 }}>{v}</Text> },
    { title: '意图', dataIndex: 'intent_tag', width: 90,
      render: (v) => v ? <Tag>{v}</Tag> : <Text type="secondary">无</Text> },
    { title: '工具', dataIndex: 'tools_used', width: 190,
      render: (v) => (v || []).join(', ') || <Text type="secondary">无</Text> },
    { title: '命中', dataIndex: 'hits', width: 60 },
    { title: '降级', dataIndex: 'degraded', width: 64,
      render: (v) => v ? <Tag color="orange">降级</Tag> : null },
    { title: '兜底', dataIndex: 'fallback', width: 64,
      render: (v) => v ? <Tag color="purple">兜底</Tag> : null },
    { title: '耗时', dataIndex: 'total_ms', width: 90,
      render: (v) => `${v}ms` },
    { title: '状态', dataIndex: 'llm_ok', width: 90,
      render: (v, r) => {
        const hasError = !v || (r.steps || []).some(s => s.status === 'error')
        return hasError ? <Tag color="red">错误</Tag>
          : (r.degraded || r.fallback) ? <Tag color="orange">降级</Tag>
          : <Tag color="green">正常</Tag>
      } },
  ]

  return (
    <div style={{ padding: 16, height: '100%', overflow: 'auto' }}>
      {/* 概览卡片 */}
      <Row gutter={12} style={{ marginBottom: 16 }}>
        <Col span={4}><Card size="small"><Statistic title="总请求" value={summary?.total ?? '-'} /></Card></Col>
        <Col span={4}><Card size="small"><Statistic title="错误" value={summary?.errors ?? '-'}
          valueStyle={{ color: (summary?.errors ?? 0) > 0 ? '#cf1322' : undefined }} /></Card></Col>
        <Col span={4}><Card size="small"><Statistic title="降级" value={summary?.degraded ?? '-'} /></Card></Col>
        <Col span={4}><Card size="small"><Statistic title="决策兜底" value={summary?.fallback ?? '-'} /></Card></Col>
        <Col span={4}><Card size="small"><Statistic title="LLM 失败" value={summary?.llm_errors ?? '-'}
          valueStyle={{ color: (summary?.llm_errors ?? 0) > 0 ? '#cf1322' : undefined }} /></Card></Col>
        <Col span={4}><Card size="small"><Statistic title="平均耗时(ms)" value={summary?.avg_ms ?? '-'} /></Card></Col>
      </Row>

      {/* 工具栏 */}
      <Space style={{ marginBottom: 12 }}>
        <Select value={status} onChange={setStatus} style={{ width: 130 }}
          options={[
            { value: 'all', label: '全部请求' },
            { value: 'error', label: '仅错误' },
            { value: 'degraded', label: '降级/兜底' },
          ]} />
        <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>刷新</Button>
        <Button icon={<SyncOutlined />}
          type={autoRefresh ? 'primary' : 'default'}
          onClick={() => setAutoRefresh(v => !v)}>
          {autoRefresh ? '自动刷新中(5s)' : '自动刷新'}
        </Button>
      </Space>

      {summary && summary.total === 0 && (
        <Alert type="info" showIcon style={{ marginBottom: 12 }}
          message="暂无监控数据——发起一轮对话后，这里会展示该请求的全链路时间线（记忆/增强/意图/工具/回写/回复）。" />
      )}

      {/* 最近请求表格 */}
      <Table rowKey="id" size="small" loading={loading}
        columns={columns} dataSource={items} pagination={{ pageSize: 20, showSizeChanger: false }}
        onRow={(record) => ({ onClick: () => openDetail(record.id), style: { cursor: 'pointer' } })} />

      {/* 请求详情：全链路时间线 */}
      <Drawer title={detail ? `请求详情 ${detail.id}` : '请求详情'}
        open={!!detail} onClose={() => setDetail(null)} width={520}>
        {detail && (
          <div>
            <Space direction="vertical" style={{ width: '100%' }} size={4}>
              <Text strong>原始问题：</Text><Text>{detail.query}</Text>
              {detail.enhanced_query !== detail.query && (
                <><Text strong>增强后问题：</Text><Text type="secondary">{detail.enhanced_query}</Text></>
              )}
              <Text strong>会话 ID：</Text><Text type="secondary">{detail.session_id || '（无）'}</Text>
              <Space wrap>
                <Tag>意图：{detail.intent_tag || '无'}</Tag>
                <Tag>工具：{detail.tool || '无'}</Tag>
                <Tag>命中：{detail.hits}</Tag>
                {detail.degraded && <Tag color="orange">降级</Tag>}
                {detail.fallback && <Tag color="purple">决策兜底</Tag>}
                {detail.context_reset && <Tag color="blue">会话过期</Tag>}
                <Tag color={detail.llm_ok ? 'green' : 'red'}>
                  LLM {detail.llm_ok ? 'OK' : '失败'}
                </Tag>
                <Tag>总耗时 {detail.total_ms}ms</Tag>
              </Space>
            </Space>

            <Typography.Title level={5} style={{ marginTop: 20 }}>
              全链路时间线
            </Typography.Title>
            <Timeline
              items={(detail.steps || []).map((s) => ({
                color: STATUS_COLOR[s.status] || 'gray',
                children: (
                  <div>
                    <Space>
                      <Text strong>{s.stage}</Text>
                      <Tag color={STATUS_COLOR[s.status]}>{s.status}</Tag>
                      <Text type="secondary">{s.ms}ms</Text>
                    </Space>
                    {s.detail && <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{s.detail}</div>}
                    {s.extra?.error && (
                      <div style={{ fontSize: 12, color: '#cf1322', marginTop: 2 }}>{s.extra.error}</div>
                    )}
                  </div>
                ),
              }))}
            />
          </div>
        )}
      </Drawer>
    </div>
  )
}
