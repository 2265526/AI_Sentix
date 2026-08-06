import React, { useState } from 'react'
import { Layout, Menu, Typography } from 'antd'
import {
  MessageOutlined,
  SearchOutlined,
  ToolOutlined,
} from '@ant-design/icons'
import ChatPage from './pages/ChatPage.jsx'
import RagPage from './pages/RagPage.jsx'
import AdminPage from './pages/AdminPage.jsx'

const { Header, Sider, Content } = Layout

const PAGES = {
  chat: { key: 'chat', label: '客服聊天', icon: <MessageOutlined />, node: <ChatPage /> },
  rag: { key: 'rag', label: 'RAG 检索验证', icon: <SearchOutlined />, node: <RagPage /> },
  admin: { key: 'admin', label: '管理员', icon: <ToolOutlined />, node: <AdminPage /> },
}

export default function App() {
  const [current, setCurrent] = useState('chat')

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider theme="dark" width={200}>
        <div style={{ color: '#fff', padding: 16, fontWeight: 600, fontSize: 15 }}>
          电商AI智能客服
          <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.45)', marginTop: 4 }}>
            功能验证台（无需登录）
          </div>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[current]}
          items={Object.values(PAGES).map(({ key, label, icon }) => ({ key, label, icon }))}
          onClick={({ key }) => setCurrent(key)}
        />
      </Sider>
      <Layout>
        <Header style={{ background: '#fff', padding: '0 24px', borderBottom: '1px solid #f0f0f0' }}>
          <Typography.Title level={4} style={{ margin: '14px 0' }}>
            {PAGES[current].label}
          </Typography.Title>
        </Header>
        <Content style={{ padding: 16, background: '#f5f5f5' }}>
          {PAGES[current].node}
        </Content>
      </Layout>
    </Layout>
  )
}
