# SceneHarbor

本地 AI 视频修复工作台，适合旧视频、剪辑归档和素材再发布。一条片子可能同时有清楚的画面、低分辨率插入镜头和压缩严重的段落。SceneHarbor 先按镜头跑模型候选，再把同一帧的多个结果摆给你看。你确认或调整后，软件在本机 GPU 上渲染整片。

## 安装

使用 Linux x86-64、Python 3.11+、带 libx264 的 FFmpeg 6+ 和可用的 Vulkan 显卡驱动。

```sh
uv tool install 'git+https://github.com/iyuzhang/sceneharbor.git@v1.0.0'
sceneharbor install-engine
sceneharbor doctor
sceneharbor serve
```

浏览器打开 **http://127.0.0.1:8752**。也可从 [GitHub Release](https://github.com/iyuzhang/sceneharbor/releases) 下载 wheel 并安装。安装引擎后可离线使用，无需 API key。

## 怎么用

1. 选择本机视频，设好输出高度和画质。大文件可直接填绝对路径，省时也省磁盘。
2. 分析完成后点开一个镜头，拖动分隔线看细节。可选原始缩放、35% AI 混合和全 AI，界面会显示建议及诊断依据。建议优先复查脸、文字、细线和运动画面。
3. 点击导出。视频较长可以暂停后续跑。后来改了一两个镜头，软件会复用其他已完成的段落。导出有 MP4 和记录每段处理方案的 JSON。

首次打开可以用 [公开样片](https://media.w3.org/2010/05/sintel/trailer.mp4)。想跑无人值守批处理时，用 `sceneharbor batch --height 720 a.mp4 b.mp4`。默认工作区在 `~/.local/share/sceneharbor/workspace`，也可用 `--workspace` 自定。

## 适用范围

支持逐行、固定帧率、SDR 视频，先把 HDR、隔行或可变帧率转成上述格式。导出保留源帧率、显示比例、音轨和章节，MP4 不兼容的音频会转为 AAC。需要字幕和容器附件时，请保留源文件。

Real-ESRGAN 逐帧推理，可能改变原来的纹理并产生闪烁。诊断分数适合定位要复查的镜头，建议保留原片。[对照实验与复现步骤](benchmark.md) 含固定缩放、固定 AI 和按镜头选择。
