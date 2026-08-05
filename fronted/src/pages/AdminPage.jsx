import React, { useState } from 'react'
import {
  Alert, Card, Divider, Select, Space, Statistic, Tabs, Typography, Upload, message,
} from 'antd'
import { InboxOutlined } from '@ant-design/icons'
import { uploadKb, importProducts } from '../api.js'

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

// ---------- 管理页 ----------
export default function AdminPage() {
  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <Alert
        type="warning"
        showIcon
        style={{ marginBottom: 16 }}
        message="管理员功能（无需登录，仅供本地验证）"
        description="本页用于知识库文档管理与商品数据同步，数据直接写入本地 PostgreSQL。"
      />
      <Card>
        <Tabs
          defaultActiveKey="kb"
          items={[
            { key: 'kb', label: '知识库文档管理', children: <KbUpload /> },
            { key: 'product', label: '商品数据同步（CSV）', children: <ProductImport /> },
          ]}
        />
      </Card>
    </div>
  )
}
