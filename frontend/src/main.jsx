import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import App from './App.jsx'

// 注意：不用 React.StrictMode——开发模式下它会双调用 setState updater，
// 配合流式追加内容（副作用式更新）会导致 token 重复（如"您好您好"）。
ReactDOM.createRoot(document.getElementById('root')).render(
  <ConfigProvider locale={zhCN}>
    <App />
  </ConfigProvider>
)
