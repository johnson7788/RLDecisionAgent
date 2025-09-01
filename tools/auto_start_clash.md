# 开机启动脚本
vim /etc/systemd/system/clash.service

[Unit]
Description=Clash Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=clash
Group=clash
WorkingDirectory=/root/clash
# 按你的二进制与目录调整：
ExecStart=/root/clash/clash-linux-amd64-v1.10.0 -d /root/clash
Restart=always
RestartSec=3
LimitNOFILE=1048576
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true

[Install]
WantedBy=multi-user.target

# 配置
sudo systemctl daemon-reload
sudo systemctl enable --now clash.service
systemctl status clash.service
journalctl -u clash.service -b
