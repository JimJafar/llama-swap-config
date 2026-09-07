// pm2 ecosystem for markdown-rag, relocated to the ZOTAC (GPU-af5bd53f).
// Relocated 2026-09-01 so it stops holding 500 MiB on the 5070 Ti, freeing VRAM
// for the vLLM Qwen3.8-27B entry. Env preserved from the previous launch.
module.exports = {
	apps: [
		{
			name: "markdown-rag",
			script: "/home/jim/.local/share/pipx/venvs/markdown-rag/bin/markdown-rag",
			interpreter: "/home/jim/.local/share/pipx/venvs/markdown-rag/bin/python",
			args: "serve /home/jim/Documents/obsidian-md --port 8123 --host 0.0.0.0",
			cwd: "/home/jim",
			autorestart: true,
			max_restarts: 10,
			restart_delay: 5000,
			env: {
				// Pin to the ZOTAC 5060 Ti so the 5070 Ti (vLLM host) stays free.
				CUDA_VISIBLE_DEVICES: "GPU-af5bd53f-a1ca-d807-1ae2-7f5ef261fe1a",
				CUDA_PATH: "/opt/cuda",
				NVCC_CCBIN: "/usr/bin/g++-15",
				HF_HOME: "/mnt/shared/models",
				PYTHONUNBUFFERED: "1",
				OLLAMA_FLASH_ATTENTION: "1",
				OLLAMA_KV_CACHE_TYPE: "q8_0",
				LD_LIBRARY_PATH:
					"/home/jim/.local/share/pipx/venvs/markdown-rag/lib/python3.14/site-packages/nvidia/cublas/lib:/home/jim/.local/share/pipx/venvs/markdown-rag/lib/python3.14/site-packages/nvidia/cuda_nvrtc/lib:/home/jim/.local/share/pipx/venvs/markdown-rag/lib/python3.14/site-packages/nvidia/cudnn/lib:/home/jim/.local/share/pipx/venvs/markdown-rag/lib/python3.14/site-packages/nvidia/cufft/lib:/home/jim/.local/share/pipx/venvs/markdown-rag/lib/python3.14/site-packages/nvidia/nvjitlink/lib",
				CONTEXT7_API_KEY: process.env.CONTEXT7_API_KEY || "",
				DEEPSEEK_API_KEY: process.env.DEEPSEEK_API_KEY || "",
				DEEPSEEK_API_URL:
					process.env.DEEPSEEK_API_URL || "https://api.deepseek.com/v1",
				LIBRARIAN_AGENT_TOKEN: process.env.LIBRARIAN_AGENT_TOKEN || "",
				LIBRARIAN_MCP_URL:
					process.env.LIBRARIAN_MCP_URL ||
					"https://ssd-nodes.akita-betelgeuse.ts.net:3839/mcp",
			},
		},
	],
};
