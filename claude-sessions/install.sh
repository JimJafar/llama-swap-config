#!/bin/sh
# Install claude-sessions on Marvin (here) and Colin, and enable it at boot on both.
#   ./install.sh            (re-run after editing sessions.txt or the script)
set -eu
cd "$(dirname "$0")"
install_here() {
    install -Dm755 claude-sessions "$HOME/.local/bin/claude-sessions"
    install -Dm644 sessions.txt "$HOME/.config/claude-sessions.txt"
    install -Dm644 claude-sessions.service "$HOME/.config/systemd/user/claude-sessions.service"
    systemctl --user daemon-reload
    systemctl --user enable claude-sessions.service
}
install_here
for h in colin; do
    ssh "$h" 'mkdir -p ~/.local/bin ~/.config/systemd/user'
    scp -q claude-sessions "$h:.local/bin/claude-sessions"
    scp -q sessions.txt "$h:.config/claude-sessions.txt"
    scp -q claude-sessions.service "$h:.config/systemd/user/claude-sessions.service"
    ssh "$h" 'chmod 755 ~/.local/bin/claude-sessions && systemctl --user daemon-reload && systemctl --user enable claude-sessions.service'
done
echo "installed; start now with: systemctl --user start claude-sessions (here and on colin)"
