# 可移植的 MiniMax H3 Skills

本目录包含 1 个通用提示词技能和 8 个创作技能，改编自
[MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills)。
来源版本、保留文件和改编范围见 [H3-SOURCES.md](H3-SOURCES.md)。

## 技能清单

| Skill | 适用任务 |
| --- | --- |
| [h3-prompt-writing](h3-prompt-writing/SKILL.md) | T2VA、I2VA、FL2VA、L2VA、Ref2VA 结构化提示词 |
| [minimalist-product-ad-generator](minimalist-product-ad-generator/SKILL.md) | 极简产品广告、独立产品参考图、文案与节奏 |
| [3d-animation-short-generator](3d-animation-short-generator/SKILL.md) | 3D 故事短片、角色/场景设计、六列镜头表、分镜与剪辑 |
| [papercraft-stop-motion-explainer](papercraft-stop-motion-explainer/SKILL.md) | 分层纸艺、纸偶、机关和微缩场景科普 |
| [brand-promo-video-generator](brand-promo-video-generator/SKILL.md) | 品牌宣传、真实素材与卖点、产品交互、CTA |
| [music-video-subtitle-generator](music-video-subtitle-generator/SKILL.md) | MV、纯音乐画面叙事、节拍编排、可选歌词排版 |
| [co-op-game-intro-generator](co-op-game-intro-generator/SKILL.md) | 双人游戏菜单、确认图与进入游戏世界的动画 |
| [paper-collage-explainer-generator](paper-collage-explainer-generator/SKILL.md) | 半色调照片剪贴、纸片逐件拼装、视觉隐喻 |
| [handdrawn-live-video-generator](handdrawn-live-video-generator/SKILL.md) | 实拍与粗糙发光手绘融合、接触、连续变形和延迟追拍 |

## 使用

Codex 可从仓库的 `.agents/skills/` 发现技能。会话启动时读取技能目录的
客户端可能需要重新打开会话。Claude Code 可通过 `.claude/skills/` 中的同名
相对符号链接读取同一份文件。其他支持 Markdown 技能的 agent 可以直接加载
对应 `SKILL.md`，并按其中指引读取参考文件。

示例：

```text
用 $music-video-subtitle-generator 为这段纯音乐设计 30 秒的二维流体 MG。
不要歌词或字幕，保持向左运动，先只输出分镜和三段 H3 提示词。
```

```text
用 $co-op-game-intro-generator 制作双人游戏开场。
玩家是 Lin 和 Mei，纸雕风，标题是 Together。
先做确认图，等我确认后再生成视频。
```

## MV smoke test

用一首带时间轴歌词的用户自有歌曲和一张原创成年人物参考图，测试
`music-video-subtitle-generator` 与 `h3-prompt-writing` 的组合流程：

```text
用 $music-video-subtitle-generator 和 $h3-prompt-writing，把这首歌设计成
古装男女爱情故事 MV。剧情表演和女主演唱交叉推进，不显示歌词文字；
按后端实际单段时长拆分，并在最终成片中完整保留原歌曲。
```

验收时检查：全局时间线与源音频等长；歌词保持原文；各生成窗口记录入场、
出场和局部时间；Ref2VA 提示词保持六段字段顺序且引用标签全部有真实输入；
拼接成片可以完整解码，并确认最终音轨来自用户提供的源文件而不是模型重建。

迁移到其他项目或用户级技能目录时，一起复制这九个技能目录，并保持相邻结构。
八个创作技能共享 `h3-prompt-writing/references/portable-workflow.md` 和 H3
格式指南；不能只复制一个 `SKILL.md`。各技能的 `references/` 与 `agents/`
也必须随目录保留。`agents/openai.yaml` 仅提供 Codex UI 元数据，不限制其他
agent 使用；工作流程不要求安装专属插件。

## 执行边界

写提示词、分镜和规划只需要能读写文件的 agent。实际图片、视频、配音、配乐
和剪辑使用当前环境已经可用的工具或服务，不假定任何未提供的工具存在。
缺少能力时，交付完整的可用制作资料并明确未完成的生成环节。

用户的风格、语言、素材、音频、模型选择和已有授权优先。不会因为套用 MV
模板就给纯音乐添加歌词，也不会给无描边 MG 添加胶片颗粒。只请求提示词时
不会启动生成；明确要求确认图后再渲染时保留该确认节点。

vLLM-Omni 的长视频扩展须以当前分支和运行后端为准。本目录不会把开发分支
的 continuation 参数当作所有 H3 服务都支持的通用 API。技能支持保存真实
请求和原始输出，区分模型结果、后期修复与超分，并检查实际音视频属性。

这些技能不包含模型权重，不会自动启动服务或调用付费生成接口。
