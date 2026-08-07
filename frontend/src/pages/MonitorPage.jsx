import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Divider,
  Drawer,
  Dropdown,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Timeline,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { DownloadOutlined, ReloadOutlined, SyncOutlined } from '@ant-design/icons'
import { exportMonitorLog, getMonitorRequest, getMonitorRequests, getMonitorSummary } from '../api.js'

const { Text } = Typography

// 状态 → 颜色/文案
const STATUS_COLOR = { ok: 'green', degraded: 'orange', error: 'red', skipped: 'default' }

const STAGE_LABEL = {
  memory: '记忆读取/增强',
  intent: '意图识别',
  tool: '工具执行',
  save: '记忆回写',
  reply: '二次回调',
}

export default function MonitorPage() {
  const [summary, setSummary] = useState(null)
  const [items, setItems] = useState([])
  const [status, setStatus] = useState('all')
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [exporting, setExporting] = useState(false)
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

  const doExport = useCallback(async ({ key }) => {
    setExporting(true)
    try {
      const name = await exportMonitorLog({ status, format: key })
      message.success(`已导出 ${name}`)
    } catch (e) {
      message.error(`导出失败: ${e.message}`)
    } finally {
      setExporting(false)
    }
  }, [status])

  const exportMenu = {
    items: [
      { key: 'csv', label: 'CSV（表格，Excel 可直接打开）' },
      { key: 'json', label: 'JSON（完整结构化）' },
      { key: 'txt', label: 'TXT（日志风格，按请求分块）' },
    ],
    onClick: doExport,
  }

  const columns = [
    { title: '时间', dataIndex: 'ts', width: 150 },
    { title: '问题', dataIndex: 'query', ellipsis: true, width: 240,
      render: (v) => <Text style={{ fontSize: 12 }}>{v}</Text> },
    { title: '意图', dataIndex: 'intent_tag', width: 90,
      render: (v) => v ? <Tag>{v}</Tag> : <Text type="secondary">无</Text> },
    { title: '工具', dataIndex: 'tools_used', width: 190,
      render: (v) => (v || []).join(', ') || <Text type="secondary">无</Text> },
    { title: '命中', dataIndex: 'hits', width: 60 },
    { title: 'Tokens', dataIndex: 'total_tokens', width: 90,
      render: (v, r) => (v || (r.prompt_tokens ?? 0) + (r.completion_tokens ?? 0)) > 0
        ? <Tooltip title={`输入 ${r.prompt_tokens ?? 0} / 输出 ${r.completion_tokens ?? 0}`}>
            <Text>{v ?? (r.prompt_tokens ?? 0) + (r.completion_tokens ?? 0)}</Text>
          </Tooltip>
        : <Text type="secondary">—</Text> },
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
        <Dropdown menu={exportMenu} disabled={exporting}>
          <Button icon={<DownloadOutlined />} loading={exporting}
            disabled={!summary || summary.total === 0}>
            导出日志
          </Button>
        </Dropdown>
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
        open={!!detail} onClose={() => setDetail(null)} width={560}>
        {detail && (
          <div>
            {/* 基本信息 */}
            <Descriptions column={1} size="small" bordered
              labelStyle={{ width: 110, fontWeight: 600 }}>
              <Descriptions.Item label="时间">{detail.ts}</Descriptions.Item>
              <Descriptions.Item label="会话 ID">{detail.session_id || '（无）'}</Descriptions.Item>
              <Descriptions.Item label="原始问题">{detail.query}</Descriptions.Item>
              <Descriptions.Item label="增强后问题">
                {detail.enhanced_query ? (
                  detail.enhanced_query === detail.query
                    ? <Text type="secondary">（与原始问题一致）</Text>
                    : detail.enhanced_query
                ) : '（无）'}
              </Descriptions.Item>
              <Descriptions.Item label="预分类意图">
                <Tag>{detail.intent_tag || '无'}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="意图工具">
                {detail.intent_tool ? <Tag>{detail.intent_tool}</Tag> : <Text type="secondary">无（模型未选中工具）</Text>}
                <div style={{ fontSize: 12, color: '#888', marginTop: 2 }}>意图识别阶段模型选中的工具（可能未实际执行）</div>
              </Descriptions.Item>
              <Descriptions.Item label="实际工具">
                {(detail.tool || (detail.tools_used || []).length > 0)
                  ? <Tag color="blue">{detail.tool || detail.tools_used.join(', ')}</Tag>
                  : <Text type="secondary">无（未执行工具）</Text>}
                <div style={{ fontSize: 12, color: '#888', marginTop: 2 }}>
                  {detail.tools_used?.length > 0
                    ? `实际执行：${detail.tools_used.join(', ')}${detail.fallback ? '（决策兜底触发）' : ''}`
                    : '决策兜底/容错后真正执行的工具'}
                </div>
              </Descriptions.Item>
              <Descriptions.Item label="Token 消耗">
                <Tag>输入 {detail.prompt_tokens ?? 0}</Tag>
                <Tag>输出 {detail.completion_tokens ?? 0}</Tag>
                <Tag color="geekblue">合计 {detail.total_tokens ?? 0}</Tag>
                <div style={{ fontSize: 12, color: '#888', marginTop: 2 }}>
                  意图识别 + 二次回调两段 LLM 调用合计
                </div>
              </Descriptions.Item>
            </Descriptions>

            {/* 状态标记 */}
            <Space wrap style={{ marginTop: 12 }}>
              <Tag>命中：{detail.hits}</Tag>
              {detail.degraded && <Tag color="orange">降级命中</Tag>}
              {detail.fallback && <Tag color="purple">决策兜底</Tag>}
              {detail.context_reset && <Tag color="blue">会话过期</Tag>}
              <Tag color={detail.llm_ok ? 'green' : 'red'}>
                LLM {detail.llm_ok ? 'OK' : '失败'}
              </Tag>
              <Tag>总耗时 {detail.total_ms}ms</Tag>
            </Space>

            {/* 工具调用参数 */}
            {detail.tool_inputs?.length > 0 && (
              <>
                <Typography.Title level={5} style={{ marginTop: 20 }}>
                  工具调用参数
                </Typography.Title>
                {detail.tool_inputs.map((t, i) => (
                  <Card key={i} size="small" style={{ marginBottom: 8 }}
                    title={<Tag color="blue">{t.name}</Tag>}>
                    <pre style={{ margin: 0, fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                      {JSON.stringify(t.arguments ?? {}, null, 2)}
                    </pre>
                  </Card>
                ))}
              </>
            )}

            {/* 工具返回摘要 */}
            {detail.tool_results_summary?.length > 0 && (
              <>
                <Typography.Title level={5} style={{ marginTop: 20 }}>
                  工具返回摘要
                </Typography.Title>
                {detail.tool_results_summary.map((t, i) => (
                  <div key={i} style={{ marginBottom: 8 }}>
                    <Tag color="blue">{t.name}</Tag> 命中 <Text strong>{t.hits}</Text> 条
                    {t.preview && (
                      <div style={{ fontSize: 12, color: '#555', marginTop: 2,
                        background: '#fafafa', padding: '6px 8px', borderRadius: 4 }}>
                        {t.preview}
                      </div>
                    )}
                  </div>
                ))}
              </>
            )}

            {/* 最终回复 */}
            <Typography.Title level={5} style={{ marginTop: 20 }}>
              最终回复{!detail.llm_ok && <Tag color="red" style={{ marginLeft: 8 }}>兜底话术</Tag>}
            </Typography.Title>
            <div style={{ background: '#fafafa', padding: '8px 12px', borderRadius: 4 }}>
              <Text>{detail.reply || '（无回复）'}</Text>
            </div>

            {/* 全链路时间线 */}
            <Typography.Title level={5} style={{ marginTop: 20 }}>
              全链路时间线
            </Typography.Title>
            <Timeline
              items={(detail.steps || []).map((s) => ({
                color: STATUS_COLOR[s.status] || 'gray',
                children: (
                  <div>
                    <Space>
                      <Text strong>{STAGE_LABEL[s.stage] || s.stage}</Text>
                      <Tag color={STATUS_COLOR[s.status]}>{s.status}</Tag>
                      <Text type="secondary">{s.ms}ms</Text>
                    </Space>
                    {s.detail && <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{s.detail}</div>}
                    {s.extra?.error && (
                      <div style={{ fontSize: 12, color: '#cf1322', marginTop: 2 }}>{s.extra.error}</div>
                    )}
                    {s.extra?.raw && (
                      <div style={{ fontSize: 12, color: '#888', marginTop: 2, wordBreak: 'break-all' }}>
                        LLM 响应: {s.extra.raw}
                      </div>
                    )}
                    {s.extra?.hits !== undefined && (
                      <div style={{ fontSize: 12, color: '#888', marginTop: 2 }}>召回 {s.extra.hits} 条</div>
                    )}
                  </div>
                ),
              }))}
            />

            <Divider />
            <Text type="secondary" style={{ fontSize: 12 }}>
              数据源：内存环形缓冲（最近 200 条，服务重启后清空）。需要留档可在列表页工具栏「导出日志」下载 CSV/JSON/TXT。
            </Text>
          </div>
        )}
      </Drawer>
    </div>
  )
}
