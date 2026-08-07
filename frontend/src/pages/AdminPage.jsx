import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert, Button, Card, Divider, Form, Input, Radio, Select, Space, Statistic,
  Table, Tabs, Tag, Tree, Typography, Upload, message,
} from 'antd'
import { InboxOutlined, PlusOutlined, SearchOutlined } from '@ant-design/icons'
import { uploadKb, importProducts, getCategories, searchCategories, createCategory } from '../api.js'

const { Dragger } = Upload

// ---------- 知识库上传 ----------
function KbUpload() {
  const [docType, setDocType] = useState('policy')
  const [result, setResult] = useState(null)

  const beforeUpload = async (file) => {
    const isSupported = /\.(txt|md|pdf)$/i.test(file.name)
    if (!isSupported) {
      message.error('仅支持 TXT / MD / PDF 文件')
      return Upload.LIST_IGNORE
    }
    if (file.size > 20 * 1024 * 1024) {
      message.error('文件超过 20MB 限制')
      return Upload.LIST_IGNORE
    }
    try {
      const data = await uploadKb(file, docType)
      setResult(data)
      message.success(`上传成功：${data.chunk_count} 个知识分块已向量化入库`)
    } catch (e) {
      message.error(`上传失败：${e.response?.data?.detail || e.message}`)
    }
    return false // 阻止 antd 默认上传
  }

  return (
    <div>
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="上传 TXT / PDF 文档，系统自动分块（500字/块，重叠50字）并调用 embedding 向量化入库，上传后立即可被 RAG 检索。"
      />
      <Space style={{ marginBottom: 12 }}>
        <span>文档类型：</span>
        <Select
          value={docType}
          onChange={setDocType}
          style={{ width: 160 }}
          options={[
            { value: 'policy', label: '政策' },
            { value: 'faq', label: '常见问题 FAQ' },
            { value: 'product_manual', label: '商品说明书' },
          ]}
        />
      </Space>
      <Dragger
        multiple={false}
        beforeUpload={beforeUpload}
        accept=".txt,.md,.pdf"
        showUploadList={false}
      >
        <p className="ant-upload-drag-icon"><InboxOutlined /></p>
        <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
        <p className="ant-upload-hint">支持 TXT / MD / PDF，单个文件 ≤ 20MB</p>
      </Dragger>
      {result && (
        <Card size="small" style={{ marginTop: 16 }} title="上传结果">
          <Statistic title="知识分块数" value={result.chunk_count} suffix="块" />
          <Typography.Text type="secondary">
            文件：{result.filename}｜类型：{result.doc_type}｜文档：{result.doc_count} 篇
          </Typography.Text>
        </Card>
      )}
    </div>
  )
}

// ---------- 商品 CSV 导入 ----------
function ProductImport() {
  const [result, setResult] = useState(null)

  const beforeUpload = async (file) => {
    if (!/\.csv$/i.test(file.name)) {
      message.error('仅支持 CSV 文件')
      return Upload.LIST_IGNORE
    }
    try {
      const data = await importProducts(file)
      setResult(data)
      if (data.failed > 0) {
        message.warning(`导入完成：成功 ${data.imported} 条，失败 ${data.failed} 条`)
      } else {
        message.success(`导入成功：${data.imported} 条商品已同步`)
      }
    } catch (e) {
      message.error(`导入失败：${e.response?.data?.detail || e.message}`)
    }
    return false
  }

  return (
    <div>
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="上传商品 CSV（列：id、title、final_price、stock、Product Description），按 SKU 增量同步商品与库存；同时自动为每个商品生成知识库向量（商品名+描述 → 分块 → 向量化，中文直接入库，不做翻译）。"
      />
      <Dragger multiple={false} beforeUpload={beforeUpload} accept=".csv" showUploadList={false}>
        <p className="ant-upload-drag-icon"><InboxOutlined /></p>
        <p className="ant-upload-text">点击或拖拽 CSV 文件到此处上传</p>
        <p className="ant-upload-hint">列名：id, title, final_price, stock, Product Description</p>
      </Dragger>
      {result && (
        <Card size="small" style={{ marginTop: 16 }} title="导入结果">
          <Space size="large">
            <Statistic title="总行数" value={result.total} />
            <Statistic title="成功导入" value={result.imported} valueStyle={{ color: '#3f8600' }} />
            <Statistic title="失败" value={result.failed} valueStyle={{ color: result.failed ? '#cf1322' : undefined }} />
            <Statistic title="生成知识文档" value={result.kb_docs_created ?? 0} />
            <Statistic title="知识分块" value={result.kb_chunks_created ?? 0} />
          </Space>
          {result.errors?.length > 0 && (
            <Typography.Paragraph type="danger" style={{ marginTop: 8, fontSize: 12 }}>
              {result.errors.join('；')}
            </Typography.Paragraph>
          )}
        </Card>
      )}
    </div>
  )
}

// ---------- 类目管理（V2.2.3）----------
const LEVEL_META = { 1: { label: '大类', color: 'blue' }, 2: { label: '中类', color: 'purple' }, 3: { label: '小类', color: 'default' } }

function CategoryManager() {
  const [tree, setTree] = useState([])
  const [loading, setLoading] = useState(false)
  // 搜索
  const [searchQ, setSearchQ] = useState('')
  const [searchResult, setSearchResult] = useState(null)
  const [searching, setSearching] = useState(false)
  // 新增表单
  const [level, setLevel] = useState(1)
  const [bigId, setBigId] = useState(null)
  const [midId, setMidId] = useState(null)
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getCategories()
      setTree(data.tree || [])
    } catch (e) {
      message.error(`类目加载失败：${e.response?.data?.detail || e.message}`)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  // 树 → antd Tree 数据（大类 → 中类 → 小类）
  const treeData = useMemo(() => tree.map(n => ({
    key: n.id,
    title: n.name,
    children: (n.children || []).map(c => ({
      key: c.id,
      title: c.name,
      children: (c.children || []).map(s => ({ key: s.id, title: s.name })),
    })),
  })), [tree])

  const bigOptions = tree.map(n => ({ value: n.id, label: n.name }))
  const midOptions = (bigId
    ? (tree.find(n => n.id === bigId)?.children || [])
    : []).map(n => ({ value: n.id, label: n.name }))

  const handleSearch = useCallback(async (q) => {
    const kw = (q || '').trim()
    if (!kw) { setSearchResult(null); return }
    setSearching(true)
    try {
      const data = await searchCategories(kw)
      setSearchResult(data)
    } catch (e) {
      message.error(`搜索失败：${e.response?.data?.detail || e.message}`)
    } finally {
      setSearching(false)
    }
  }, [])

  const handleSubmit = async () => {
    const n = (name || '').trim()
    if (!n) { message.warning('请输入类目名称'); return }
    let payload
    if (level === 1) {
      payload = { name: n, level: 1, parent_id: null }
    } else if (level === 2) {
      if (!bigId) { message.warning('请先选择所属大类'); return }
      payload = { name: n, level: 2, parent_id: bigId }
    } else {
      if (!bigId) { message.warning('请先选择所属大类'); return }
      if (!midId) { message.warning('请先选择所属中类'); return }
      payload = { name: n, level: 3, parent_id: midId }
    }
    setSubmitting(true)
    try {
      await createCategory(payload)
      message.success(`已新增${LEVEL_META[level].label}：${n}`)
      setName('')
      setMidId(null)
      await load()
    } catch (e) {
      message.error(e.response?.data?.detail || e.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div>
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="类目为三级结构（大类 → 中类 → 小类）。新增中类需先选所属大类，新增小类需先选所属大类与中类；搜索支持类目名称或完整路径的模糊/相似匹配。"
      />

      {/* 类目搜索 */}
      <Card size="small" title="类目搜索" style={{ marginBottom: 16 }}>
        <Input.Search
          placeholder="输入类目名或路径关键词，如：衬衫 / 手机 / 服装鞋包/女装"
          allowClear
          enterButton={<><SearchOutlined /> 搜索</>}
          value={searchQ}
          onChange={(e) => { setSearchQ(e.target.value); if (!e.target.value.trim()) setSearchResult(null) }}
          onSearch={handleSearch}
          loading={searching}
          style={{ maxWidth: 520 }}
        />
        {searchResult && (
          <Table
            rowKey="id"
            size="small"
            style={{ marginTop: 12 }}
            pagination={false}
            dataSource={searchResult.items || []}
            locale={{ emptyText: `未找到与「${searchResult.query}」相关的类目` }}
            columns={[
              { title: '类目名称', dataIndex: 'name', width: 180, render: (v) => <Typography.Text strong>{v}</Typography.Text> },
              { title: '层级', dataIndex: 'level', width: 80,
                render: (v) => <Tag color={LEVEL_META[v]?.color}>{LEVEL_META[v]?.label}</Tag> },
              { title: '完整路径', dataIndex: 'path', render: (v) => <Typography.Text type="secondary">{v}</Typography.Text> },
            ]}
          />
        )}
      </Card>

      {/* 新增类目 */}
      <Card size="small" title="新增类目" style={{ marginBottom: 16 }}>
        <Space wrap align="end">
          <div>
            <div style={{ marginBottom: 4 }}>层级</div>
            <Radio.Group value={level} onChange={(e) => { setLevel(e.target.value); setBigId(null); setMidId(null) }}
              options={[
                { value: 1, label: '大类' },
                { value: 2, label: '中类' },
                { value: 3, label: '小类' },
              ]} />
          </div>
          {level >= 2 && (
            <div>
              <div style={{ marginBottom: 4 }}>所属大类（必选）</div>
              <Select
                showSearch optionFilterProp="label" style={{ width: 200 }}
                placeholder="选择大类" value={bigId || undefined}
                onChange={(v) => { setBigId(v); setMidId(null) }}
                options={bigOptions} />
            </div>
          )}
          {level === 3 && (
            <div>
              <div style={{ marginBottom: 4 }}>所属中类（必选）</div>
              <Select
                showSearch optionFilterProp="label" style={{ width: 200 }}
                placeholder="选择中类" value={midId || undefined} onChange={setMidId}
                options={midOptions} />
            </div>
          )}
          <div>
            <div style={{ marginBottom: 4 }}>类目名称</div>
            <Input
              style={{ width: 200 }} placeholder={`输入${LEVEL_META[level].label}名称`}
              value={name} onChange={(e) => setName(e.target.value)}
              onPressEnter={handleSubmit} maxLength={50} />
          </div>
          <Button type="primary" icon={<PlusOutlined />} loading={submitting} onClick={handleSubmit}>
            新增{LEVEL_META[level].label}
          </Button>
        </Space>
      </Card>

      {/* 类目树 */}
      <Card size="small" title={`类目结构（共 ${tree.length} 个大类）`} loading={loading}>
        {tree.length === 0 ? (
          <Typography.Text type="secondary">暂无类目数据</Typography.Text>
        ) : (
          <Tree
            showLine
            defaultExpandAll={false}
            defaultExpandedKeys={tree.map(n => n.key)}
            treeData={treeData}
            style={{ maxHeight: 420, overflow: 'auto' }}
          />
        )}
      </Card>
    </div>
  )
}

// ---------- 管理页 ----------
export default function AdminPage() {
  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <Alert
        type="warning"
        showIcon
        style={{ marginBottom: 16 }}
        message="管理员功能（无需登录，仅供本地验证）"
        description="本页用于知识库文档管理、商品数据同步与平台类目维护，数据直接写入本地 PostgreSQL。"
      />
      <Card>
        <Tabs
          defaultActiveKey="kb"
          items={[
            { key: 'kb', label: '知识库文档管理', children: <KbUpload /> },
            { key: 'product', label: '商品数据同步（CSV）', children: <ProductImport /> },
            { key: 'category', label: '类目管理', children: <CategoryManager /> },
          ]}
        />
      </Card>
    </div>
  )
}
