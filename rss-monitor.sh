#!/usr/bin/env bash
# Samples llama-swap RSS + pm2 restart count every 5 min into a bounded log.
# Purpose: determine whether the ~90MB -> ~330MB growth under load is an
# unbounded leak or a plateau (Go-heap retention) before deciding on GOMEMLIMIT.
set -u
LOG=~/llama-swap/memory-monitor.log
STATE=~/.local/state/llama-swap-ssm.offset
INTERVAL=300

sample() {
	local line
	line=$(
		python3 - <<'EOF' 2>/dev/null
import json, datetime, os, subprocess
out = subprocess.run(["pm2", "jlist"], capture_output=True, text=True).stdout
data = json.loads(out or "[]")
for p in data:
    if p.get("name") == "llama-swap":
        pm = p.get("pm2_env", {})
        rss = p.get("monit", {}).get("memory", 0)
        # uptime from /proc: starttime (field 22, clock ticks) vs /proc/uptime
        up = "?"
        try:
            clk = os.sysconf("SC_CLK_TCK")
            up_now = float(open("/proc/uptime").read().split()[0])
            start = int(open(f"/proc/{p['pid']}/stat").read().rsplit(")", 1)[1].split()[19])
            up = int((up_now - start / clk) // 60)
        except Exception:
            pass
        print(
            datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
            + f" rss={rss/1048576:.1f}MB restarts={pm.get('restart_time', 0)} "
            + f"uptime={up}m pid={p.get('pid')}"
        )
        break
EOF
	)
	if [ -n "$line" ]; then
		echo "$line" >>"$LOG"
		# Append any new max-memory-restart events from pm2.log (tracked by offset)
		local pm2log=~/.pm2/pm2.log last=0
		if [ -f "$pm2log" ]; then last=$(grep -c "" "$pm2log" || echo 0); fi
		local prev=$(cat "$STATE" 2>/dev/null || echo 0)
		if [ "$last" -gt "$prev" ] && [ "$prev" -gt 0 ]; then
			tail -n +$((prev + 1)) "$pm2log" | grep "max-memory-restart" >>"$LOG" 2>/dev/null || true
		fi
		[ "$last" -gt 0 ] && echo "$last" >"$STATE"
		# Keep the log bounded: trim to last 2000 lines past ~500KB
		if [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 512000 ]; then
			tail -n 2000 "$LOG" >"$LOG.tmp" && mv "$LOG.tmp" "$LOG"
		fi
	fi
}

sample
while true; do
	sleep "$INTERVAL"
	sample
done
