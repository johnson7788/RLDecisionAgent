# 开机启动脚本
vim /etc/systemd/system/llmcache.service

[Unit]
Description=LLM Cache Service
After=network-online.target clash.service
Requires=clash.service    # 如果强依赖代理，确保 clash 异常退出时本服务也停

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/root/RLDecisionAgent/backend/ART_mcp-rl
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/vipuser/miniconda3/bin/python /root/RLDecisionAgent/backend/ART_mcp-rl/LLM_cache.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

# 配置
sudo systemctl daemon-reload
sudo systemctl enable --now llmcache.service
systemctl status llmcache.service
journalctl -u llmcache.service -b