import React, { useState } from 'react'
import { Layout, Menu, Typography } from 'antd'
import {
  MessageOutlined,
  SearchOutlined,
  ToolOutlined,
  MonitorOutlined,
} from '@ant-design/icons'
import ChatPage from './pages/ChatPage.jsx'
import RagPage from './pages/RagPage.jsx'
import AdminPage from './pages/AdminPage.jsx'
import MonitorPage from './pages/MonitorPage.jsx'

const { Header, Sider, Content } = Layout

const PAGES = {
  chat: { key: 'chat', label: '客服聊天', icon: <MessageOutlined />, node: <ChatPage /> },
  rag: { key: 'rag', label: 'RAG 检索验证', icon: <SearchOutlined />, node: <RagPage /> },
  monitor: { key: 'monitor', label: '监控', icon: <MonitorOutlined />, node: <MonitorPage /> },
  admin: { key: 'admin', label: '管理员', icon: <ToolOutlined />, node: <AdminPage /> },
}

export default function App() {
  const [current, setCurrent] = useState('chat')

  return (
    <Layout style={{ height: '100vh', overflow: 'hidden' }}>
      <Sider theme="dark" width={200}>
        <div style={{ color: '#fff', padding: '12px 16px 8px', fontWeight: 600, fontSize: 15, lineHeight: 1.4 }}>
          电商AI智能客服
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[current]}
          items={Object.values(PAGES).map(({ key, label, icon }) => ({ key, label, icon }))}
          onClick={({ key }) => setCurrent(key)}
          style={{ borderInlineEnd: 'none' }}
        />
      </Sider>
      <Layout style={{ height: '100vh' }}>
        <Header style={{ background: '#fff', padding: '0 24px', borderBottom: '1px solid #f0f0f0', height: 56, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography.Title level={4} style={{ margin: 0 }}>
            {PAGES[current].label}
          </Typography.Title>
        </Header>
        <Content style={{ padding: 0, background: '#f5f5f5', overflow: 'hidden' }}>
          {PAGES[current].node}
        </Content>
      </Layout>
    </Layout>
  )
}
