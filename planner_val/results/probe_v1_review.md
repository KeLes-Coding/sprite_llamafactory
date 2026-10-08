# probe_v1 逐题明细

## 来源

- **作者与审核**：107 道题由助手手写，未经人工审核。题目文件目前只存在于 `/tmp/probe_v1.json`，尚未归档。
- **出题方法**：取 11 个种子句（f1–f10 是单任务，f11 是多任务结构），每道题只改动一个维度：换话术、换问法、换参数、换结构，另有反例。
- **标准答案**：按 v3.4.0 训练口径编写。
- **评分口径**：严格分。在 benchmark 规则之上，还要求 step、degree、city、query_type 完全一致。
- **锚点**：qwen-flash 使用线上提示词。离线映射只做两件事：去掉天气查询的经纬度；把 repeat 展开成多个任务。
- **结果目录**：
  - `gemma3-270m-lora-v340-probe_v1-20261008-144750`
  - `qwen3-0.6b-lora-v340-probe_v1-20261008-144523`
  - `qwen-flash-probe_v1-20261008-145956`
- **生成方式**：本文件由 `/tmp/probe_dump.py` 生成。

## 有争议的标准答案（10 题）

下面这些题的标准答案只是 v3.4.0 的口径约定，换一种理解也说得通：

| id | 输入 | 本集标准答案 | 另一种合理理解 |
|---|---|---|---|
| f1_06 | 鞠躬，来个三连。 | 鞠躬 ×3 | 鞠躬后再接其他动作 |
| f1_07 | 鞠躬这个礼节在日本是什么意思？ | reply | web_search |
| f3_05 | 原地向左转两圈。 | 一次转 720° | 转两次 360°，或回复超出能力 |
| f5_03 | 躺平吧，摆烂一会儿。 | lie_down | “躺平”是俚语，可以只回复 |
| f5_08 | 小明，你过来这边坐下。 | reply（叫的是别人） | 如果机器人就叫小明，应当执行 |
| f8_03 | 雷克雅未克明天会下雪吗？ | 24h | 7d |
| f8_06 | 查下阿勒泰现在的天气和接下来七天的。 | 一次 now_7d | 先查 now，再查 7d（语义等价） |
| f8_09 | 我在纠结杭州明天穿啥，看看天气呗。 | 24h | now_7d |
| f11_19 | 如果待会下雨你就跳舞，现在先查一下广州今晚会不会下雨。 | reply → 天气 | 只查天气 |
| f11_20 | 能先帮我查下深圳现在的天气吗？再跳个舞。 | 默认 warm_up_dance | 任选一支舞 |

锚点还有两题判错，原因是线上 schema 和 v3.4.0 不一致，离线映射没有覆盖：

- **f3_05**：锚点输出 `step=2` 表示转两圈，v3.4.0 要求转身时 `step=1`。
- **f11_16**：锚点输出 `x=3`，按米计，v3.4.0 用的是方向 × 步数。

### 得分

| 口径 | 270M | 0.6B | 锚点 |
|---|---|---|---|
| 全部 107 题 | 83 | 89 | 93 |
| 去掉 10 道争议题（97 题） | 77 | 83 | 90 |

97 题口径下，0.6B 与锚点的配对检验：0.6B 错而锚点对 13 题，锚点错而 0.6B 对 6 题，McNemar p = 0.17，差异不显著。

## 逐题结果

✓ / ✗ 均为严格分。

| id | 维度 | 输入 | 标准答案 | 270M | 0.6B | 锚点 |
|---|---|---|---|---|---|---|
| f1_01 | 种子 | 给大家鞠个躬吧。 | 手势:bow | ✓ | ✓ | ✓ |
| f1_02 | 换话术 | 来来来，冲在座各位老师深深地弯个腰，表示一下敬意。 | 手势:bow | ✗ | ✓ | ✓ |
| f1_03 | 换话术 | Take a bow for the folks here, would ya. | 手势:bow | ✓ | ✓ | ✓ |
| f1_04 | 换问法 | 客人都到了，还不快鞠个躬？ | 手势:bow | ✓ | ✓ | ✓ |
| f1_05 | 换问法 | 我想看你给大家鞠一躬。 | 手势:bow | ✓ | ✓ | ✓ |
| f1_06 | 换参数 | 鞠躬，来个三连。 | 手势:bow → 手势:bow → 手势:bow | ✗ | ✗ | ✗ |
| f1_07 | 反例 | 鞠躬这个礼节在日本是什么意思？ | reply | ✓ | ✓ | ✗ |
| f1_08 | 反例 | 等主持人讲完话，你再鞠躬。 | reply | ✗ | ✓ | ✓ |
| f2_01 | 种子 | 往前走三步。 | walk(x=0.6,y=0,yaw=0,step=3) | ✓ | ✓ | ✓ |
| f2_02 | 换话术 | 劳驾您老往前头挪三步呗，别挡着后面的人。 | walk(x=0.6,y=0,yaw=0,step=3) | ✓ | ✓ | ✓ |
| f2_03 | 换话术 | Scoot up three paces. | walk(x=0.6,y=0,yaw=0,step=3) | ✗ | ✓ | ✗ |
| f2_04 | 换问法 | 我想让你往前走三步。 | walk(x=0.6,y=0,yaw=0,step=3) | ✓ | ✓ | ✓ |
| f2_05 | 换问法 | 你往前走三步好不好？ | walk(x=0.6,y=0,yaw=0,step=3) | ✓ | ✓ | ✓ |
| f2_06 | 换参数 | 往前走十七步。 | walk(x=0.6,y=0,yaw=0,step=17) | ✓ | ✓ | ✓ |
| f2_07 | 换参数 | 向前走 12 步。 | walk(x=0.6,y=0,yaw=0,step=12) | ✓ | ✓ | ✓ |
| f2_08 | 换参数 | 往前挪个二十五步。 | walk(x=0.6,y=0,yaw=0,step=25) | ✓ | ✓ | ✓ |
| f2_09 | 换参数 | Walk forward eight steps. | walk(x=0.6,y=0,yaw=0,step=8) | ✓ | ✓ | ✓ |
| f2_10 | 换结构 | 前进，三步。 | walk(x=0.6,y=0,yaw=0,step=3) | ✗ | ✓ | ✓ |
| f3_01 | 种子 | 向左转九十度。 | walk(x=0,y=0,yaw=1,step=1,degree=90) | ✓ | ✓ | ✓ |
| f3_02 | 换话术 | 往左手边拧过去九十度。 | walk(x=0,y=0,yaw=1,step=1,degree=90) | ✗ | ✓ | ✓ |
| f3_03 | 换参数 | 向左转七十五度。 | walk(x=0,y=0,yaw=1,step=1,degree=75) | ✓ | ✓ | ✓ |
| f3_04 | 换参数 | 向右转两百度。 | walk(x=0,y=0,yaw=-1,step=1,degree=200) | ✗ | ✗ | ✓ |
| f3_05 | 换参数 | 原地向左转两圈。 | walk(x=0,y=0,yaw=1,step=1,degree=720) | ✗ | ✗ | ✗ |
| f3_06 | 换参数 | 向右转一百一十度。 | walk(x=0,y=0,yaw=-1,step=1,degree=110) | ✓ | ✓ | ✓ |
| f3_07 | 换参数 | 往右转个 40 度。 | walk(x=0,y=0,yaw=-1,step=1,degree=40) | ✓ | ✓ | ✓ |
| f3_08 | 换参数 | Turn left 270 degrees. | walk(x=0,y=0,yaw=1,step=1,degree=270) | ✓ | ✓ | ✓ |
| f3_09 | 换参数 | Rotate right by 100 degrees. | walk(x=0,y=0,yaw=-1,step=1,degree=100) | ✓ | ✓ | ✓ |
| f3_10 | 换问法 | 你朝左转九十度试试。 | walk(x=0,y=0,yaw=1,step=1,degree=90) | ✓ | ✓ | ✓ |
| f4_01 | 种子 | 往右边平移两步。 | walk(x=0,y=-0.6,yaw=0,step=2) | ✓ | ✓ | ✓ |
| f4_02 | 换话术 | 学螃蟹那样，往右横着走四步。 | walk(x=0,y=-0.6,yaw=0,step=4) | ✗ | ✓ | ✓ |
| f4_03 | 换参数 | 往左横移九步。 | walk(x=0,y=0.6,yaw=0,step=9) | ✓ | ✓ | ✓ |
| f4_04 | 换参数 | Sidestep to the left five steps. | walk(x=0,y=0.6,yaw=0,step=5) | ✓ | ✗ | ✓ |
| f5_01 | 种子 | 坐下吧。 | sit_down() | ✓ | ✓ | ✓ |
| f5_02 | 换话术 | 累了吧？一屁股坐下歇会儿。 | sit_down() | ✓ | ✓ | ✓ |
| f5_03 | 换话术 | 躺平吧，摆烂一会儿。 | lie_down() | ✓ | ✓ | ✓ |
| f5_04 | 换话术 | 起立！ | stand_up() | ✓ | ✓ | ✓ |
| f5_05 | 换话术 | Have a seat. | sit_down() | ✗ | ✓ | ✓ |
| f5_06 | 换话术 | Take a load off for a minute. | sit_down() | ✗ | ✗ | ✗ |
| f5_07 | 换问法 | 你干嘛还站着，坐呀。 | sit_down() | ✗ | ✓ | ✓ |
| f5_08 | 反例 | 小明，你过来这边坐下。 | reply | ✗ | ✗ | ✗ |
| f6_01 | 种子 | 跳个机械舞。 | 舞:popping | ✓ | ✓ | ✓ |
| f6_02 | 换话术 | 整一段机械舞，嗨起来！ | 舞:popping | ✓ | ✓ | ✓ |
| f6_03 | 换话术 | popping 走一个。 | 舞:popping | ✓ | ✓ | ✓ |
| f6_04 | 换问法 | 我想看你跳机械舞。 | 舞:popping | ✓ | ✓ | ✓ |
| f6_05 | 换问法 | 不如来段机械舞热热场？ | 舞:popping | ✓ | ✓ | ✓ |
| f6_06 | 换参数 | 来一段卡啦永远OK。 | 舞:karla_ok | ✓ | ✓ | ✓ |
| f6_07 | 换参数 | 跳低俗小说那个舞。 | 舞:pulp_fiction_dance | ✓ | ✓ | ✓ |
| f6_08 | 换参数 | 跳个相亲相爱。 | 舞:luv_each_other | ✓ | ✓ | ✓ |
| f6_09 | 换参数 | 埃及摇摆来一个。 | 舞:egyptian_shake | ✓ | ✓ | ✗ |
| f6_10 | 换参数 | 来段 Gee。 | 舞:gee | ✓ | ✓ | ✓ |
| f6_11 | 反例 | 我昨天看人跳了段机械舞，挺帅的。 | reply | ✓ | ✓ | ✓ |
| f6_12 | 反例 | If I win this round, you dance for me, deal? | reply | ✓ | ✓ | ✓ |
| f7_01 | 种子 | 挥挥手跟大家打个招呼。 | 手势:wave_greet_bye | ✓ | ✓ | ✓ |
| f7_02 | 换参数 | 用左手比个心。 | 手势:left_hand_side_heart | ✓ | ✓ | ✓ |
| f7_03 | 换参数 | 右手侧着比个心。 | 手势:right_hand_side_heart | ✓ | ✓ | ✓ |
| f7_04 | 换参数 | 来个谢幕动作。 | 手势:curtain_bow | ✓ | ✓ | ✓ |
| f7_05 | 换参数 | 给大家飞几个吻。 | 手势:blow_kisses_multi | ✓ | ✓ | ✓ |
| f7_06 | 换参数 | 做个请这边走的手势。 | 手势:this_way_please | ✓ | ✓ | ✓ |
| f7_07 | 换参数 | 跟这位先生握个手。 | 手势:shake_hands | ✓ | ✗ | ✓ |
| f7_08 | 换话术 | 冲大伙儿摆摆手，打个招呼。 | 手势:wave_greet_bye | ✓ | ✓ | ✓ |
| f7_09 | 换话术 | Give everyone a little wave. | 手势:wave_greet_bye | ✓ | ✓ | ✓ |
| f7_10 | 反例 | 我家狗都会握手，你会吗？ | reply | ✓ | ✓ | ✓ |
| f8_01 | 种子 | 北京今天天气怎么样？ | 天气(北京,now) | ✓ | ✓ | ✓ |
| f8_02 | 换参数 | 喀什现在多少度？ | 天气(喀什,now) | ✓ | ✗ | ✓ |
| f8_03 | 换参数 | 雷克雅未克明天会下雪吗？ | 天气(雷克雅未克,24h) | ✗ | ✗ | ✗ |
| f8_04 | 换参数 | 未来一周漠河冷不冷？ | 天气(漠河,7d) | ✗ | ✗ | ✓ |
| f8_05 | 换参数 | What's the weather like in Tbilisi this week? | 天气(Tbilisi,7d) | ✓ | ✓ | ✓ |
| f8_06 | 换参数 | 查下阿勒泰现在的天气和接下来七天的。 | 天气(阿勒泰,now_7d) | ✓ | ✓ | ✗ |
| f8_07 | 换话术 | 老家驻马店今晚下雨不？ | 天气(驻马店,24h) | ✓ | ✓ | ✓ |
| f8_08 | 换话术 | 瞅瞅上海外头这会儿啥天儿。 | 天气(上海,now) | ✗ | ✗ | ✓ |
| f8_09 | 换问法 | 我在纠结杭州明天穿啥，看看天气呗。 | 天气(杭州,24h) | ✓ | ✓ | ✗ |
| f9_01 | 种子 | 你还剩多少电？ | 状态:电量 | ✓ | ✓ | ✓ |
| f9_02 | 换话术 | 电池还剩几格？ | 状态:电量 | ✓ | ✓ | ✓ |
| f9_03 | 换话术 | How much juice have you got left? | 状态:电量 | ✗ | ✓ | ✓ |
| f9_04 | 换参数 | 你现在在哪儿呢？ | 状态:位置 | ✓ | ✓ | ✓ |
| f9_05 | 换参数 | 今儿几号？ | 状态:日期 | ✓ | ✓ | ✓ |
| f9_06 | 换参数 | 现在几点钟了？ | 状态:时间 | ✓ | ✓ | ✓ |
| f9_07 | 换参数 | 你是哪个型号的？ | 状态:型号 | ✓ | ✓ | ✓ |
| f10_01 | 种子 | 帮我搜一下今天的科技新闻。 | web_search(query=x) | ✓ | ✓ | ✓ |
| f10_02 | 换参数 | 今年诺贝尔文学奖给了谁？ | web_search(query=x) | ✗ | ✗ | ✓ |
| f10_03 | 换参数 | 最近油价涨了没？ | web_search(query=x) | ✓ | ✓ | ✓ |
| f10_04 | 换参数 | 华为 Mate 70 现在卖多少钱？ | web_search(query=x) | ✓ | ✓ | ✓ |
| f10_05 | 换参数 | Look up the latest iPhone price. | web_search(query=x) | ✓ | ✓ | ✓ |
| f10_06 | 换参数 | Luna 的保修期多长？ | rag_query(query=x) | ✓ | ✓ | ✓ |
| f10_07 | 换参数 | TRON1 能爬多高的台阶？ | rag_query(query=x) | ✓ | ✓ | ✓ |
| f10_08 | 换参数 | 逐际动力总部在哪？ | rag_query(query=x) | ✓ | ✓ | ✓ |
| f10_09 | 反例 | 你说我该不该辞职？ | reply | ✓ | ✗ | ✓ |
| f11_01 | 换结构 | 1. 鞠躬 2. 往前走两步 3. 查一下北京现在的天气 | 手势:bow → walk(x=0.6,y=0,yaw=0,step=2) → 天气(北京,now) | ✓ | ✗ | ✓ |
| f11_02 | 换结构 | - wave hello - turn right 90 degrees - check battery | 手势:wave_greet_bye → walk(x=0,y=0,yaw=-1,step=1,degree=90) → 状态:电量 | ✓ | ✓ | ✓ |
| f11_03 | 换结构 | 我跟你说啊，今天早上地铁挤死了，差点没挤上去。你先挥个手跟大家打个招呼。然后我同事说中午吃火锅，我是吃不了辣的，算了不说这个。你再往左转四十五度。 | 手势:wave_greet_bye → walk(x=0,y=0,yaw=1,step=1,degree=45) | ✓ | ✓ | ✓ |
| f11_04 | 换结构 | 左转右转各九十度。 | walk(x=0,y=0,yaw=1,step=1,degree=90) → walk(x=0,y=0,yaw=-1,step=1,degree=90) | ✗ | ✗ | ✓ |
| f11_05 | 换结构 | 依次鞠躬、点头、挥手。 | 手势:bow → 手势:nod → 手势:wave_greet_bye | ✓ | ✓ | ✓ |
| f11_06 | 换结构 | 跳个机械舞，跳完再来一遍。 | 舞:popping → 舞:popping | ✗ | ✗ | ✓ |
| f11_07 | 换结构 | 鞠躬、点头、挥手，除了点头都做一下。 | 手势:bow → 手势:wave_greet_bye | ✗ | ✗ | ✓ |
| f11_08 | 换结构 | 你知道吗，我第一次来这个展会，啥也不懂，听说你们公司挺厉害的，那你先给我跳个热场舞吧。 | 舞:warm_up_dance | ✓ | ✓ | ✗ |
| f11_09 | 换结构 | 往前走（大概五步吧）然后坐下。 | walk(x=0.6,y=0,yaw=0,step=5) → sit_down() | ✓ | ✓ | ✓ |
| f11_10 | 换结构 | 👋 打个招呼，然后 💃 跳个APT。 | 手势:wave_greet_bye → 舞:apt_dance | ✓ | ✓ | ✗ |
| f11_11 | 换结构 | 先站起来，然后往前走两步，然后右转九十度，然后再往前走三步，然后坐下，然后告诉我现在几点。 | stand_up() → walk(x=0.6,y=0,yaw=0,step=2) → walk(x=0,y=0,yaw=-1,step=1,degree=90) → walk(x=0.6,y=0,yaw=0,step=3) → sit_down() → 状态:时间 | ✓ | ✓ | ✓ |
| f11_12 | 换结构 | Stand up, walk forward 4 steps, turn left 180 degrees, do the popping dance, nod, check the weather in Paris right now, and tell me your battery level. | stand_up() → walk(x=0.6,y=0,yaw=0,step=4) → walk(x=0,y=0,yaw=1,step=1,degree=180) → 舞:popping → 手势:nod → 天气(Paris,now) → 状态:电量 | ✓ | ✓ | ✓ |
| f11_13 | 换结构 | 第一，挥手；第二，鞠躬；第三，跳个胜利之舞。 | 手势:wave_greet_bye → 手势:bow → 舞:victory_dance | ✓ | ✓ | ✓ |
| f11_14 | 换结构 | First 鞠躬, then 往右转 sixty degrees, finally check 上海 weather tomorrow. | 手势:bow → walk(x=0,y=0,yaw=-1,step=1,degree=60) → 天气(上海,24h) | ✗ | ✓ | ✓ |
| f11_15 | 换结构 | 北京和上海现在的天气都查一下。 | 天气(北京,now) → 天气(上海,now) | ✗ | ✗ | ✓ |
| f11_16 | 换结构 | 去前面三步那儿，到了之后坐下。 | walk(x=0.6,y=0,yaw=0,step=3) → sit_down() | ✗ | ✓ | ✗ |
| f11_17 | 换结构 | 坐下之前，先往后退两步。 | walk(x=-0.6,y=0,yaw=0,step=2) → sit_down() | ✓ | ✓ | ✗ |
| f11_18 | 换结构 | 你先往前走两步再左转九十度然后鞠个躬最后查一下北京明天的天气 | walk(x=0.6,y=0,yaw=0,step=2) → walk(x=0,y=0,yaw=1,step=1,degree=90) → 手势:bow → 天气(北京,24h) | ✓ | ✓ | ✓ |
| f11_19 | 换结构 | 如果待会下雨你就跳舞，现在先查一下广州今晚会不会下雨。 | reply → 天气(广州,24h) | ✓ | ✓ | ✓ |
| f11_20 | 换结构 | 能先帮我查下深圳现在的天气吗？再跳个舞。 | 天气(深圳,now) → 舞:warm_up_dance | ✓ | ✓ | ✓ |

合计（严格分）：270M 83/107，0.6B 89/107，锚点 93/107

## 错题的模型原始输出

| id | 模型 | 模型输出 | 判错原因 |
|---|---|---|---|
| f1_02 | 270M | reply | tool_sequence_mismatch |
| f1_06 | 270M | 手势:bow → 舞:warm_up_dance → 舞:warm_up_dance → 舞:warm_up_dance | forbidden_tool |
| f1_06 | 0.6B | 手势:bow → 舞:warm_up_dance | forbidden_tool |
| f1_06 | 锚点 | 手势:bow → 手势:high_five → 手势:high_five → 手势:high_five | forbidden_tool |
| f1_07 | 锚点 | web_search(query=鞠躬这个礼节在日本是什么意思？) | forbidden_tool |
| f1_08 | 270M | 手势:bow | forbidden_tool |
| f2_03 | 270M | 舞:scoot_up_three_paces | parser_rejected |
| f2_03 | 锚点 | walk(x=0,y=1,yaw=0,step=3) | parameter_mismatch |
| f2_10 | 270M | walk(x=0.6,y=0,yaw=0,step=1) → walk(x=0.6,y=0,yaw=0,step=3) | forbidden_tool |
| f3_02 | 270M | walk(x=0,y=1,yaw=0,step=90) | parser_rejected |
| f3_04 | 270M | walk(x=0,y=0,yaw=-1,degree=21,step=1) | degree 21!=200 |
| f3_04 | 0.6B | walk(x=0,y=0,yaw=-1,degree=180,step=1) | degree 180!=200 |
| f3_05 | 270M | walk(x=0,y=0,yaw=1,degree=24,step=1) | degree 24!=720 |
| f3_05 | 0.6B | walk(x=0,y=0,yaw=1,degree=360,step=1) | degree 360!=720 |
| f3_05 | 锚点 | walk(x=0,y=0,yaw=1,degree=720,step=2) | parser_rejected（schema 差异） |
| f4_02 | 270M | reply | tool_sequence_mismatch |
| f4_04 | 0.6B | sit_down() → walk(x=0,y=1,yaw=0,step=5) | forbidden_tool |
| f5_05 | 270M | 手势:curtain_bow | forbidden_tool |
| f5_06 | 270M | walk(x=0,y=1,yaw=0,step=1) | parameter_mismatch |
| f5_06 | 0.6B | rest() | parser_rejected |
| f5_06 | 锚点 | reply | tool_sequence_mismatch |
| f5_07 | 270M | 状态:当前动作 → sit_down() | forbidden_tool |
| f5_08 | 270M | sit_down() | forbidden_tool |
| f5_08 | 0.6B | sit_down() | forbidden_tool |
| f5_08 | 锚点 | sit_down() | forbidden_tool |
| f6_09 | 锚点 | 舞:abracadabr_dance | parameter_mismatch |
| f7_07 | 0.6B | reply | tool_sequence_mismatch |
| f8_02 | 0.6B | reply | tool_sequence_mismatch |
| f8_03 | 270M | 天气(深圳,24h) | city 深圳!=雷克雅未克 |
| f8_03 | 0.6B | 天气(深圳,24h) | city 深圳!=雷克雅未克 |
| f8_03 | 锚点 | 天气(雷克雅未克,7d) | qt 7d!=24h |
| f8_04 | 270M | 天气(漠河,now) | qt now!=7d |
| f8_04 | 0.6B | 天气(深圳,7d) | city 深圳!=漠河 |
| f8_06 | 锚点 | 天气(阿勒泰,now) → 天气(阿勒泰,7d) | forbidden_tool |
| f8_08 | 270M | 天气(上海外头这会儿,None) | parser_rejected |
| f8_08 | 0.6B | 状态:日期 | forbidden_tool |
| f8_09 | 锚点 | 天气(杭州,now_7d) → reply | qt now_7d!=24h |
| f9_03 | 270M | 状态:状态 | parser_rejected |
| f10_02 | 270M | reply ×9 | tool_sequence_mismatch |
| f10_02 | 0.6B | reply | tool_sequence_mismatch |
| f10_09 | 0.6B | web_search(query=该不该辞职) | forbidden_tool |
| f11_01 | 0.6B | 手势:bow → walk(step=1) → walk(step=1) → 天气(北京,now) | forbidden_tool |
| f11_04 | 270M | walk(x=0,y=0,yaw=-1,degree=90,step=1) | tool_sequence_mismatch |
| f11_04 | 0.6B | walk(yaw=1,degree=90) → walk(yaw=1,degree=90) | parameter_mismatch |
| f11_06 | 270M | 舞:popping | tool_sequence_mismatch |
| f11_06 | 0.6B | 舞:popping | tool_sequence_mismatch |
| f11_07 | 270M | 手势:bow | tool_sequence_mismatch |
| f11_07 | 0.6B | 手势:bow → 手势:nod → 手势:wave_greet_bye | forbidden_tool |
| f11_08 | 锚点 | rag_query(query=介绍一下你们公司) → 舞:warm_up_dance | forbidden_tool |
| f11_10 | 锚点 | reply → 舞:apt_dance | tool_sequence_mismatch |
| f11_14 | 270M | 手势:bow → walk(yaw=1,degree=60) → 天气(Shanghai,24h) | parameter_mismatch |
| f11_15 | 270M | 天气(北京,now_7d) → 天气(上海,now) | qt now_7d!=now |
| f11_15 | 0.6B | 天气(北京,now_7d) → 天气(深圳,now_7d) | qt now_7d!=now; city 深圳!=上海 |
| f11_16 | 270M | 舞:warm_up_dance ×3 → sit_down() | forbidden_tool |
| f11_16 | 锚点 | walk(x=3,y=0,yaw=0,step=3) → sit_down() | parser_rejected（schema 差异：x 为米） |
| f11_17 | 锚点 | sit_down → walk(x=-1,step=2)，后面还附了一段自我解释文字 | parser_rejected（顺序错误 + 多余文本） |
